/*
 * AMBE+2 encoder: 160 samples -> 9 bytes (py/ambe_encoder.py encode_pcm with py/ambe_analysis.py, streaming).
 *
 * Frame k is analysed around input sample 160 k - 200. The pitch tracker looks two frames ahead (E(P) of frame
 * k + 2 reaches sample 160 k + 280), so frame k is encoded once frame k + 1 is in: one frame of delay.
 * Per frame: energy -> voice activity (speech or a silence frame); speech: the tracked pitch refined by a harmonic
 * fit, voicing per 500 Hz band and the harmonics' amplitudes; then the quantizer, which keeps a mirror of the
 * decoder's prediction.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "ambe.h"
#include "ambe_tables.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FS 8000
#define N AMBE_N
#define NW 256
#define NFFT 1024
#define NBINS (NFFT / 2 + 1)
#define CENTER (-200)
#define PMIN 20
#define VOICED_ERR 0.8
#define VOICING_HYSTERESIS 2.0
#define REFINE_FMAX 3000
#define QUIET_FULL (-25.0)
#define QUIET_NONE (-42.0)
#define SUBMULT_MAXERR 1.0
#define HOLD_E 0.8
#define VOICE_MAX_E 0.85
#define VOICE_E_FULL 0.4
#define VOICE_E_NONE 0.8
#define LONG_P 95
#define LONG_MAX_E 0.3
#define SUBMULT_TOL 0.05
#define HALF_ODD_DB 6.0
#define OUT_GAIN 1.7
#define NI 301
#define NP 217               /* P_GRID: 20 .. 128 in steps of 0.5 */
#define PITCH_JUMP_COST 1.0
#define TRACK_PMAX 100
#define LOWBAND_MIN 0.03
#define WRES 64              /* window transform table: NFFT * WRES points */
#define NWT (NFFT * WRES)
#define WA_BINS 64           /* the real window transform is tabulated out to this many bins */
#define WA_LEN (WA_BINS * WRES + 2)
/* voice activity */
#define VAD_MARGIN 15.0
#define VAD_MIN_DB 20.0
#define FLOOR_RISE 0.05
#define PEAK_DECAY 0.05
#define VAD_BELOW_PEAK 45.0

/* ---- constant tables, made once ---- */

static double P_GRID[NP], LOG_P[NP];
static int LO[NP], HI[NP];
static double WI2[NI], SUM_W4;
static double LPF[21];
static double WIN[NW], SUM_WIN2;
static double WA[WA_LEN];  /* the window's transform without its linear phase: real and even, WRES a bin */
static int tables_ready;

static void fft(double *re, double *im, int n)
{
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            double t = re[i];
            re[i] = re[j];
            re[j] = t;
            t = im[i];
            im[i] = im[j];
            im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2 * M_PI / len;
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; j++) {
                double wr = cos(ang * j), wi = sin(ang * j);
                double ur = re[i + j], ui = im[i + j];
                double vr = re[i + j + len / 2] * wr - im[i + j + len / 2] * wi;
                double vi = re[i + j + len / 2] * wi + im[i + j + len / 2] * wr;
                re[i + j] = ur + vr;
                im[i + j] = ui + vi;
                re[i + j + len / 2] = ur - vr;
                im[i + j + len / 2] = ui - vi;
            }
        }
    }
}

static double hamming(int i, int n)
{
    return 0.54 - 0.46 * cos(2 * M_PI * i / (n - 1));
}

static void make_tables(void)
{
    for (int i = 0; i < NP; i++) {
        P_GRID[i] = PMIN + 0.5 * i;
        LOG_P[i] = log(P_GRID[i]);
    }
    for (int j = 0; j < NP; j++) { /* searchsorted (left) of 0.8 P and 1.2 P */
        double a = 0.8 * P_GRID[j] - 1e-9, b = 1.2 * P_GRID[j] + 1e-9;
        int lo = 0, hi = 0;
        while (lo < NP && P_GRID[lo] < a) {
            lo++;
        }
        while (hi < NP && P_GRID[hi] < b) {
            hi++;
        }
        LO[j] = lo;
        HI[j] = hi;
    }
    double s2 = 0;
    for (int i = 0; i < NI; i++) {
        double w = hamming(i, NI);
        WI2[i] = w * w;
        s2 += w * w;
    }
    SUM_W4 = 0;
    for (int i = 0; i < NI; i++) {
        WI2[i] /= s2;
        SUM_W4 += WI2[i] * WI2[i];
    }
    double ls = 0;
    for (int t = -10; t <= 10; t++) {
        double x = 2.0 * 1000 / FS * t;
        double sinc = t == 0 ? 1.0 : sin(M_PI * x) / (M_PI * x);
        LPF[t + 10] = sinc * hamming(t + 10, 21);
        ls += LPF[t + 10];
    }
    for (int i = 0; i < 21; i++) {
        LPF[i] /= ls;
    }
    SUM_WIN2 = 0;
    for (int i = 0; i < NW; i++) {
        WIN[i] = hamming(i, NW);
        SUM_WIN2 += WIN[i] * WIN[i];
    }
    for (int i = 0; i < WA_LEN; i++) {
        double w = 2 * M_PI * i / NWT, a = 0;
        for (int n = 0; n < NW; n++) {
            a += WIN[n] * cos(w * (n - (NW - 1) / 2.0));
        }
        WA[i] = a;
    }
    tables_ready = 1;
}

/* ---- the harmonic fit (c/analysis.c) ---- */

static double window_at(double dw)
{
    double pos = fabs(dw) / (2 * M_PI) * NWT;
    long i = (long)pos;
    if (i >= WA_LEN - 1) {
        return 0.0;
    }
    double f = pos - i;
    return WA[i] * (1 - f) + WA[i + 1] * f;
}

/* Least-squares fit of the window's transform at l w0 to the spectrum over each harmonic's band
   (l - 0.5) w0 <= w_k < (l + 0.5) w0: A_l = sum W S / sum W^2, the band's energy en_l = sum |S|^2 and the error
   en_l - |sum W S|^2 / sum W^2. W is real: spectrum() takes the window's linear phase out of S.
   A_re, A_im, err, en: L + 1 entries each, [0] = 0 */
static void harmonic_fit(const double *s_re, const double *s_im, double w0, int L, double *A_re, double *A_im,
                         double *err, double *en)
{
    for (int l = 0; l <= L; l++) {
        A_re[l] = A_im[l] = err[l] = en[l] = 0;
    }
    for (int l = 1; l <= L; l++) {
        double lo = (l - 0.5) * w0, hi = (l + 0.5) * w0;
        int k0 = (int)floor(lo * NFFT / (2.0 * M_PI)) - 1;
        if (k0 < 0) {
            k0 = 0;
        }
        double nr = 0, ni = 0, ww = 0, e = 0;
        for (int k = k0; k < NBINS; k++) {
            double wk = ((double)k * 2.0) * M_PI / NFFT;
            if (wk >= hi) {
                break;
            }
            if (wk < lo) {
                continue;
            }
            double w = window_at(wk - l * w0);
            nr += w * s_re[k];
            ni += w * s_im[k];
            ww += w * w;
            e += s_re[k] * s_re[k] + s_im[k] * s_im[k];
        }
        if (ww <= 0) {
            continue;
        }
        A_re[l] = nr / ww;
        A_im[l] = ni / ww;
        en[l] = e;
        err[l] = e - (nr * nr + ni * ni) / ww;
    }
}

/* ---- input access ---- */

/* input sample at absolute index a (0 before the start) */
static double xin(const ambe_encoder *e, long a)
{
    long i = a - (e->n_in - AMBE_ENC_HIST);
    return i < 0 || i >= AMBE_ENC_HIST ? 0.0 : e->x[i];
}

static double xlp(const ambe_encoder *e, long a)
{
    double s = 0;
    for (int t = -10; t <= 10; t++) {
        s += LPF[t + 10] * xin(e, a + t);
    }
    return s;
}

/* the windowed spectrum of frame k (NW samples at CENTER, mean removed) */
static void spectrum(const ambe_encoder *e, long k, double *re, double *im)
{
    double s[NW], mean = 0;
    long a = N * k + CENTER - NW / 2;
    for (int i = 0; i < NW; i++) {
        s[i] = xin(e, a + i);
        mean += s[i];
    }
    mean /= NW;
    double r[NFFT], m[NFFT];
    for (int i = 0; i < NFFT; i++) {
        r[i] = i < NW ? (s[i] - mean) * WIN[i] : 0;
        m[i] = 0;
    }
    fft(r, m, NFFT);
    for (int i = 0; i < NBINS; i++) { /* times e^(j w (NW - 1) / 2): the window centred, its transform real */
        double ph = 2 * M_PI * i / NFFT * (NW - 1) / 2.0, c = cos(ph), sn = sin(ph);
        re[i] = r[i] * c - m[i] * sn;
        im[i] = r[i] * sn + m[i] * c;
    }
}

static double frame_energy_db(const ambe_encoder *e, long k)
{
    double s[NW], mean = 0, p = 0;
    long a = N * k + CENTER - NW / 2;
    for (int i = 0; i < NW; i++) {
        s[i] = xin(e, a + i);
        mean += s[i];
    }
    mean /= NW;
    for (int i = 0; i < NW; i++) {
        p += (s[i] - mean) * (s[i] - mean) * WIN[i] * WIN[i];
    }
    return 10 * log10(p / NW / (SUM_WIN2 / NW) + 1e-3);
}

/* ---- pitch tracker (py Analyzer.E, track) ---- */

static void compute_E(const ambe_encoder *e, long k, double *E)
{
    double s[NI], sw[NI], num0 = 0, raw = 0;
    long a = N * k + CENTER - NI / 2;
    for (int i = 0; i < NI; i++) {
        s[i] = xlp(e, a + i);
        double x = xin(e, a + i);
        sw[i] = s[i] * WI2[i];
        num0 += s[i] * s[i] * WI2[i];
        raw += x * x * WI2[i];
    }
    if (num0 < 1e-6 || num0 < LOWBAND_MIN * raw) {
        for (int i = 0; i < NP; i++) {
            E[i] = 1.0;
        }
        return;
    }
    double r[NI / 2 + 2];
    for (int t = 0; t <= NI / 2 + 1; t++) {
        double c = 0;
        for (int i = 0; i + t < NI; i++) {
            c += sw[i] * sw[i + t];
        }
        r[t] = c;
    }
    for (int i = 0; i < NP; i++) {
        double P = P_GRID[i], tot = 0;
        int nl = (int)((NI / 2) / P);
        for (int m = 1; m <= nl; m++) {
            double lag = P * m;
            int t = (int)floor(lag);
            double f = lag - t;
            tot += f == 0 ? r[t] : r[t] * (1 - f) + r[t + 1] * f;
        }
        tot = r[0] + 2 * tot;
        E[i] = P > TRACK_PMAX ? 1.0 : (num0 - P * tot) / (num0 * (1 - P * SUM_W4));
    }
}

/* E[0..2] = E(P) of frames k .. k + 2, reusing what the last frame computed */
static void window_E(ambe_encoder *e, long k)
{
    double tmp[3][NP];
    int have[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        long f = k + i;
        if (f >= e->E_frame && f <= e->E_frame + 2) {
            memcpy(tmp[i], e->E[f - e->E_frame], sizeof(tmp[i]));
            have[i] = 1;
        }
    }
    for (int i = 0; i < 3; i++) {
        if (have[i]) {
            memcpy(e->E[i], tmp[i], sizeof(tmp[i]));
        } else {
            compute_E(e, k + i, e->E[i]);
        }
    }
    e->E_frame = k;
}

static void best_from(const double *cost, int j, double *val, int *idx)
{
    double best = 0;
    int bi = -1;
    for (int i = LO[j]; i < HI[j]; i++) {
        double r = cost[i] + PITCH_JUMP_COST * fabs(LOG_P[i] - LOG_P[j]);
        if (bi < 0 || r < best) {
            best = r;
            bi = i;
        }
    }
    *val = best;
    if (idx) {
        *idx = bi;
    }
}

static double track(ambe_encoder *e, long k)
{
    window_E(e, k);
    const double *E0 = e->E[0], *E1 = e->E[1], *E2 = e->E[2];
    const double a = PITCH_JUMP_COST;
    int i1 = e->hist_i[0], i2 = e->hist_i[1];
    double e1 = e->hist_e[0], e2 = e->hist_e[1];
    (void)i2;
    double cB;
    int iB;
    best_from(E0, i1, &cB, &iB);
    double ceB = cB + e1 + e2;
    double c12[NP], ceF[NP];
    for (int j = 0; j < NP; j++) {
        double m2;
        best_from(E2, j, &m2, NULL);
        c12[j] = E1[j] + m2;
    }
    for (int i = 0; i < NP; i++) {
        double v;
        best_from(c12, i, &v, NULL);
        ceF[i] = E0[i] + v;
    }
    int iF = 0;
    for (int i = 1; i < NP; i++) {
        if (ceF[i] < ceF[iF]) {
            iF = i;
        }
    }
    double cF = ceF[iF];
    for (int d = (int)(P_GRID[iF] / PMIN); d > 1; d--) {
        double target = P_GRID[iF] / d;
        int j = 0;
        for (int i = 1; i < NP; i++) {
            if (fabs(P_GRID[i] - target) < fabs(P_GRID[j] - target)) {
                j = i;
            }
        }
        double c = ceF[j];
        double r = c / (cF > 1e-9 ? cF : 1e-9);
        if ((c <= 0.85 && r <= 1.7) || (c <= 0.4 && r <= 3.5) || (c <= 0.05 && r <= 5.0)) {
            iF = j;
            break;
        }
    }
    int i = (ceB <= 0.48 || ceB <= ceF[iF] + a * fabs(LOG_P[iF] - LOG_P[i1])) ? iB : iF;
    e->hist_i[1] = i1;
    e->hist_e[1] = e1;
    e->hist_i[0] = i;
    e->hist_e[0] = E0[i];
    e->track_e = E0[i];
    return P_GRID[i];
}

/* ---- pitch refinement (py Analyzer.refine, pitch) ---- */

static double fit_error(const double *sr, const double *si, double q)
{
    int L = (int)(REFINE_FMAX / ((double)FS / q));
    if (L < 1) {
        L = 1;
    }
    double ar[64], ai[64], err[64], en[64];
    harmonic_fit(sr, si, 2 * M_PI / q, L, ar, ai, err, en);
    double se = 0, sn = 0;
    for (int l = 1; l <= L; l++) {
        se += err[l];
        sn += en[l];
    }
    return se / (sn > 1e-9 ? sn : 1e-9);
}

/* tuple min of (e, q) */
static void tmin(double *e, double *q, double e2, double q2)
{
    if (e2 < *e || (e2 == *e && q2 < *q)) {
        *e = e2;
        *q = q2;
    }
}

static void refine(const double *sr, const double *si, double p, double *err, double *w0)
{
    const double span = 0.03;
    int n = 13;
    double e = 0, q = 0;
    for (int i = 0; i < n; i++) {
        double qi = p * (1 - span) + (2 * p * span) * i / (n - 1);
        double ei = fit_error(sr, si, qi);
        if (i == 0) {
            e = ei;
            q = qi;
        } else {
            tmin(&e, &q, ei, qi);
        }
    }
    double step = 0.0025 * p;
    for (int it = 0; it < 4; it++) {
        double e0 = e, q0 = q;
        double em = fit_error(sr, si, q0 - step), ep = fit_error(sr, si, q0 + step);
        tmin(&e, &q, em, q0 - step);
        tmin(&e, &q, ep, q0 + step);
        (void)e0;
        step /= 2;
    }
    double el = fit_error(sr, si, q - step), er = fit_error(sr, si, q + step);
    double den = el - 2 * e + er;
    if (den > 0) {
        double t = 0.5 * (el - er) / den;
        q += step * (t > 1 ? 1 : t < -1 ? -1 : t);
    }
    double m = e < el ? e : el;
    *err = m < er ? m : er;
    *w0 = 2 * M_PI / q;
}

static double pitch(ambe_encoder *e, long k, const double *sr, const double *si)
{
    double p = track(e, k);
    double err, w0;
    refine(sr, si, p, &err, &w0);
    int periodic = err < SUBMULT_MAXERR; /* SUBMULT_MAX_E is off */
    for (int m = 3; m >= 2 && periodic; m--) {
        double pm = 2 * M_PI / w0 / m;
        if (pm >= PMIN) {
            double ek, wk;
            refine(sr, si, pm, &ek, &wk);
            if (ek < err + SUBMULT_TOL) {
                err = ek;
                w0 = wk;
                break;
            }
        }
    }
    if (4 * M_PI / w0 <= TRACK_PMAX) {
        /* the second harmonic taken for the pitch: w0 / 2 when its odd harmonics have energy */
        double h = w0 / 2;
        int n = (int)(2000 / (FS * h / (2 * M_PI)));
        double ar[64], ai[64], er[64], en[64];
        harmonic_fit(sr, si, h, n, ar, ai, er, en);
        double odd = 0, even = 0;
        for (int l = 1; l <= n; l++) {
            double f = l * FS * h / (2 * M_PI);
            double p2 = f >= 400 ? ar[l] * ar[l] + ai[l] * ai[l] : 0;
            if (l & 1) {
                odd += p2;
            } else {
                even += p2;
            }
        }
        int near = e->have_last && fabs(log(h / e->last_w0)) < fabs(log(w0 / e->last_w0));
        if (near && even > 0 && odd > even * pow(10, -HALF_ODD_DB / 10)) {
            w0 = h;
        }
    }
    if (e->track_e > HOLD_E && e->have_last) {
        w0 = e->last_w0;
    }
    e->prev_p = 2 * M_PI / w0;
    e->last_w0 = w0;
    e->have_last = 1;
    return w0;
}

/* ---- harmonics: voicing and amplitudes (py Analyzer.harmonics) ---- */

static int nearest_b0(double w0)
{
    double lw = log(w0 / (2 * M_PI));
    int best = 0;
    double bd = 0;
    for (int i = 0; i < 120; i++) {
        double d = fabs(log((double)AmbeW0table[i]) - lw);
        if (i == 0 || d < bd) {
            bd = d;
            best = i;
        }
    }
    return best;
}

/* -> b0 of w0 and L; voiced[1..L], log2a[1..L], en[1..L] */
static int harmonics(ambe_encoder *e, const double *sr, const double *si, double w0, double level, double track_e,
                     int *voiced, double *log2a, double *en)
{
    int b0 = nearest_b0(w0);
    w0 = 2 * M_PI * AmbeW0table[b0];
    int L = AmbeLtable[b0];
    double ar[64], ai[64], err[64];
    harmonic_fit(sr, si, w0, L, ar, ai, err, en);
    double e_b[9] = {0}, n_b[9] = {0};
    int band[64];
    for (int l = 1; l <= L; l++) {
        band[l] = (int)(l * w0 / (2 * M_PI) * 16);
        if (band[l] > 8) {
            band[l] = 8;
        }
        e_b[band[l]] += err[l];
        n_b[band[l]] += en[l];
    }
    double quiet = (level - QUIET_NONE) / (QUIET_FULL - QUIET_NONE);
    quiet = quiet < 0 ? 0 : quiet > 1 ? 1 : quiet;
    if (track_e > (2 * M_PI / w0 >= LONG_P ? LONG_MAX_E : VOICE_MAX_E)) {
        quiet = 0;
    }
    double r = (VOICE_E_NONE - track_e) / (VOICE_E_NONE - VOICE_E_FULL);
    quiet *= r < 0 ? 0 : r > 1 ? 1 : r;
    int vb[8];
    for (int b = 0; b < 8; b++) {
        double thr = VOICED_ERR * quiet * (e->prev_bands[b] ? VOICING_HYSTERESIS : 1 / VOICING_HYSTERESIS);
        vb[b] = e_b[b] < thr * n_b[b];
        e->prev_bands[b] = vb[b];
    }
    for (int l = 1; l <= L; l++) {
        voiced[l] = vb[band[l] < 7 ? band[l] : 7] && en[l] > 0;
        double av = 2 * sqrt(ar[l] * ar[l] + ai[l] * ai[l]) / OUT_GAIN;
        double au = sqrt(4 * en[l] / (NFFT * SUM_WIN2)) / OUT_GAIN;
        log2a[l] = log2(voiced[l] ? (av > 1e-3 ? av : 1e-3) : (au > 1e-3 ? au : 1e-3));
    }
    return b0;
}

static void noise_amplitudes(const double *sr, const double *si, double *log2a)
{
    double ar[16], ai[16], err[16], en[16];
    harmonic_fit(sr, si, 2 * M_PI / 32, 14, ar, ai, err, en);
    for (int l = 1; l <= 14; l++) {
        double a = sqrt(4 * en[l] / (NFFT * SUM_WIN2)) / OUT_GAIN;
        log2a[l] = log2(a > 1e-3 ? a : 1e-3);
    }
}

/* ---- quantizer (py Quantizer) ---- */

static const float *hoc_table(int i, int c)
{
    switch (i) {
    case 0:
        return AmbeHOCb5[c];
    case 1:
        return AmbeHOCb6[c];
    case 2:
        return AmbeHOCb7[c];
    default:
        return AmbeHOCb8[c];
    }
}

static const int HOC_N[4] = {32, 16, 16, 8};

/* C(k) = 1/J sum_j x(j) cos(pi k (j - 0.5) / J), k = 0..J-1 */
static void dct(const double *x, int J, double *C)
{
    for (int k = 0; k < J; k++) {
        double s = 0;
        for (int j = 1; j <= J; j++) {
            s += x[j - 1] * cos(M_PI * k * (j - 0.5) / J);
        }
        C[k] = s / J;
    }
}

/* the decoder's residual T(1..L) = A24 G(2..4) + A58 G(5..8) + sum_i H_i HOC_i, every column centred (mean over
   l taken out, as the decoder takes out the mean of T) */
static void recon_matrices(int L, double A24[][4], double A58[][4], double H[4][57][4], int *hn)
{
    double CR[8][8];
    for (int i = 1; i <= 8; i++) {
        for (int m = 1; m <= 8; m++) {
            CR[i - 1][m - 1] = (m == 1 ? 1 : 2) * cos(M_PI * (m - 1) * (i - 0.5) / 8);
        }
    }
    const double rc = 1 / (2 * sqrt(2.0));
    double Cg[8][8];
    for (int i = 0; i < 4; i++) {
        for (int m = 0; m < 8; m++) {
            Cg[2 * i][m] = 0.5 * CR[2 * i][m] + 0.5 * CR[2 * i + 1][m];
            Cg[2 * i + 1][m] = rc * CR[2 * i][m] - rc * CR[2 * i + 1][m];
        }
    }
    double A[57][8];
    memset(A, 0, sizeof(A));
    memset(H, 0, sizeof(double) * 4 * 57 * 4);
    int s = 0;
    for (int i = 0; i < 4; i++) {
        int J = AmbeLmprbl[L][i];
        int n = (J < 6 ? J : 6) - 2;
        hn[i] = n > 0 ? n : 0;
        for (int jj = 1; jj <= J; jj++) {
            double idct[17];
            for (int k = 1; k <= J; k++) {
                idct[k - 1] = (k == 1 ? 1 : 2) * cos(M_PI * (k - 1) * (jj - 0.5) / J);
            }
            for (int m = 0; m < 8; m++) {
                A[s + jj - 1][m] += idct[0] * Cg[2 * i][m] + (J >= 2 ? idct[1] * Cg[2 * i + 1][m] : 0);
            }
            for (int c = 0; c < n; c++) {
                H[i][s + jj - 1][c] = idct[2 + c];
            }
        }
        s += J;
    }
    for (int l = 0; l < L; l++) {
        for (int c = 0; c < 3; c++) {
            A24[l][c] = A[l][1 + c];
        }
        A24[l][3] = 0;
        for (int c = 0; c < 4; c++) {
            A58[l][c] = A[l][4 + c];
        }
    }
    /* centre the columns */
    for (int c = 0; c < 4; c++) {
        double m24 = 0, m58 = 0, mh[4] = {0};
        for (int l = 0; l < L; l++) {
            if (c < 3) {
                m24 += A24[l][c];
            }
            m58 += A58[l][c];
            for (int i = 0; i < 4; i++) {
                mh[i] += H[i][l][c];
            }
        }
        for (int l = 0; l < L; l++) {
            if (c < 3) {
                A24[l][c] -= m24 / L;
            }
            A58[l][c] -= m58 / L;
            for (int i = 0; i < 4; i++) {
                H[i][l][c] -= mh[i] / L;
            }
        }
    }
}

/* indices of the k smallest of v[0..n), ascending */
static int smallest(const double *v, int n, int k, int *out)
{
    int m = 0;
    for (int i = 0; i < n; i++) {
        if (m < k || v[i] < v[out[m - 1]]) {
            int p = m < k ? m++ : k - 1;
            while (p > 0 && v[out[p - 1]] > v[i]) {
                out[p] = out[p - 1];
                p--;
            }
            out[p] = i;
        }
    }
    return m;
}

/* sum_l w_l x_l y_l */
static double wdot(const double *w, const double *x, const double *y, int L)
{
    double s = 0;
    for (int l = 0; l < L; l++) {
        s += (w ? w[l] : 1.0) * x[l] * y[l];
    }
    return s;
}

/* Gram matrix G = X' W X and b = X' W t of the n columns X[.][c] (stride 4) */
static void gram(const double *w, double X[][4], int n, const double *t, int L, double G[4][4], double *bv)
{
    double col[4][57];
    for (int c = 0; c < n; c++) {
        for (int l = 0; l < L; l++) {
            col[c][l] = X[l][c];
        }
    }
    for (int c = 0; c < n; c++) {
        for (int d = c; d < n; d++) {
            G[c][d] = G[d][c] = wdot(w, col[c], col[d], L);
        }
        bv[c] = wdot(w, col[c], t, L);
    }
}

/* v' G v - 2 v' b */
static double quad(double G[4][4], const double *bv, const float *v, int n)
{
    double s = 0;
    for (int c = 0; c < n; c++) {
        double g = 0;
        for (int d = 0; d < n; d++) {
            g += G[c][d] * v[d];
        }
        s += v[c] * (g - 2 * bv[c]);
    }
    return s;
}

/* PRBA and HOC by the weighted error of the reconstructed residual (py Quantizer._weighted): sum_l w_l (T_l - r_l)^2
   with the means taken out, w_l = (M_l / max M)^0.5. b3 among its 32 best unweighted entries, b4 among its 16 best,
   every pair; then each block's HOC given the rest. The errors are quadratic forms in the 3 or 4 codebook values. */
static void weighted(int *b, const double *r, const double *log2ml, int L)
{
    static double A24[57][4], A58[57][4], H[4][57][4];
    int hn[4];
    recon_matrices(L, A24, A58, H, hn);
    double mx = log2ml[1], rm = 0;
    for (int l = 1; l <= L; l++) {
        mx = log2ml[l] > mx ? log2ml[l] : mx;
        rm += r[l];
    }
    rm /= L;
    double w[57], rc[57], target[57];
    for (int l = 0; l < L; l++) {
        w[l] = pow(2.0, 0.5 * (log2ml[l + 1] - mx));
        rc[l] = r[l + 1] - rm;
        target[l] = rc[l];
    }
    for (int i = 0; i < 4; i++) {
        const float *h = hoc_table(i, b[5 + i]);
        for (int l = 0; l < L; l++) {
            target[l] -= H[i][l][0] * h[0] + H[i][l][1] * h[1] + H[i][l][2] * h[2] + H[i][l][3] * h[3];
        }
    }
    /* candidates: the best unweighted entries */
    double G[4][4], g[4], e24[512], e58[128];
    gram(NULL, A24, 3, target, L, G, g);
    for (int c = 0; c < 512; c++) {
        e24[c] = quad(G, g, AmbePRBA24[c], 3);
    }
    gram(NULL, A58, 4, target, L, G, g);
    for (int c = 0; c < 128; c++) {
        e58[c] = quad(G, g, AmbePRBA58[c], 4);
    }
    int c3[32], c4[16];
    smallest(e24, 512, 32, c3);
    smallest(e58, 128, 16, c4);
    /* every pair, weighted: |t - A24 v - A58 u|^2_w = const + q(v) + q(u) + 2 v' (A24' W A58) u */
    double Gv[4][4], gv[4], Gu[4][4], gu[4], X[4][4] = {{0}};
    gram(w, A24, 3, target, L, Gv, gv);
    gram(w, A58, 4, target, L, Gu, gu);
    for (int c = 0; c < 3; c++) {
        for (int d = 0; d < 4; d++) {
            double s = 0;
            for (int l = 0; l < L; l++) {
                s += w[l] * A24[l][c] * A58[l][d];
            }
            X[c][d] = s;
        }
    }
    double qu[16];
    for (int j = 0; j < 16; j++) {
        qu[j] = quad(Gu, gu, AmbePRBA58[c4[j]], 4);
    }
    double best = 0;
    int have = 0, b3 = 0, b4 = 0;
    for (int a = 0; a < 32; a++) {
        const float *v = AmbePRBA24[c3[a]];
        double qv = quad(Gv, gv, v, 3), xv[4];
        for (int d = 0; d < 4; d++) {
            xv[d] = 2 * (X[0][d] * v[0] + X[1][d] * v[1] + X[2][d] * v[2]);
        }
        double bj = 0;
        int jbest = -1;
        for (int j = 0; j < 16; j++) {
            const float *u = AmbePRBA58[c4[j]];
            double s = qv + qu[j] + xv[0] * u[0] + xv[1] * u[1] + xv[2] * u[2] + xv[3] * u[3];
            if (jbest < 0 || s < bj) {
                bj = s;
                jbest = j;
            }
        }
        if (!have || bj < best) {
            best = bj;
            b3 = c3[a];
            b4 = c4[jbest];
            have = 1;
        }
    }
    b[3] = b3;
    b[4] = b4;
    /* each block's HOC given the rest */
    double base[57];
    for (int l = 0; l < L; l++) {
        const float *v = AmbePRBA24[b3], *u = AmbePRBA58[b4];
        base[l] = rc[l] - (A24[l][0] * v[0] + A24[l][1] * v[1] + A24[l][2] * v[2]) -
                  (A58[l][0] * u[0] + A58[l][1] * u[1] + A58[l][2] * u[2] + A58[l][3] * u[3]);
    }
    for (int i = 0; i < 4; i++) {
        if (hn[i] == 0) {
            continue;
        }
        double rest[57];
        for (int l = 0; l < L; l++) {
            rest[l] = base[l];
        }
        for (int m = 0; m < 4; m++) {
            if (m == i) {
                continue;
            }
            const float *h = hoc_table(m, b[5 + m]);
            for (int l = 0; l < L; l++) {
                rest[l] -= H[m][l][0] * h[0] + H[m][l][1] * h[1] + H[m][l][2] * h[2] + H[m][l][3] * h[3];
            }
        }
        double Gh[4][4], gh[4], bv = 0;
        gram(w, H[i], 4, rest, L, Gh, gh);
        int bc = -1;
        for (int c = 0; c < HOC_N[i]; c++) {
            double q = quad(Gh, gh, hoc_table(i, c), 4);
            if (bc < 0 || q < bv) {
                bv = q;
                bc = c;
            }
        }
        b[5 + i] = bc;
    }
}

/* b0 given (or 124 for silence); voiced, log2ml, weight for harmonics 1..L -> 49 bits; updates the mirror */
static void quantize(ambe_encoder *e, int b0, const int *vl, const double *log2ml, const double *weight,
                     uint8_t d49[49])
{
    int b[9] = {0};
    int silence = b0 == 124;
    int L;
    double f0 = 0;
    b[0] = b0;
    if (silence) {
        b[1] = 16;
        L = 14;
    } else {
        f0 = AmbeW0table[b0];
        L = AmbeLtable[b0];
        double best = 0;
        for (int c = 0; c < 32; c++) {
            double s = 0;
            for (int l = 1; l <= L; l++) {
                s += weight[l] * (AmbeVuv[c][(int)(l * 16.0 * f0)] != vl[l]);
            }
            if (c == 0 || s < best) {
                best = s;
                b[1] = c;
            }
        }
    }
    ambe_parms *prev = &e->qprev;
    double mean = 0;
    for (int l = 1; l <= L; l++) {
        mean += log2ml[l];
    }
    double gamma = mean / L + 0.5 * log2((double)L);
    double bd = 0;
    for (int c = 0; c < 32; c++) {
        double d = fabs(AmbeDg[c] - (gamma - 0.5 * prev->gamma));
        if (c == 0 || d < bd) {
            bd = d;
            b[2] = c;
        }
    }
    /* prediction (eq. 40-43) and the residual */
    double lm[58];
    for (int l = 0; l < 58; l++) {
        lm[l] = prev->log2Ml[l];
    }
    for (int l = prev->L + 1; l <= L; l++) {
        lm[l] = lm[prev->L];
    }
    lm[0] = lm[1];
    double r[58];
    for (int l = 1; l <= L; l++) {
        double fl = ((double)prev->L / L) * l;
        int ki = (int)fl;
        double dd = fl - ki;
        r[l] = log2ml[l] - 0.65 * ((1 - dd) * lm[ki] + dd * lm[ki + 1]);
    }
    /* the HOC of each block's DCT as a start (the targets of weighted(), which picks b3, b4 and then the HOC) */
    double C[4][17], bv = 0;
    int s = 1;
    for (int i = 0; i < 4; i++) {
        int J = AmbeLmprbl[L][i];
        dct(&r[s], J, C[i]);
        s += J;
    }
    for (int i = 0; i < 4; i++) {
        int J = AmbeLmprbl[L][i];
        int n = (J < 6 ? J : 6) - 2;
        if (n <= 0) {
            continue;
        }
        for (int c = 0; c < HOC_N[i]; c++) {
            const float *h = hoc_table(i, c);
            double d = 0;
            for (int t = 0; t < n; t++) {
                d += (h[t] - C[i][2 + t]) * (h[t] - C[i][2 + t]);
            }
            if (c == 0 || d < bv) {
                bv = d;
                b[5 + i] = c;
            }
        }
    }
    weighted(b, r, log2ml, L);
    ambe_pack_b(b, d49);
    ambe_parms keep = *prev;
    ambe_decode_parms(d49, &e->qcur, prev);
    *prev = silence ? keep : e->qcur;
}

/* ---- the stream ---- */

void ambe_enc_init(ambe_encoder *e)
{
    if (!tables_ready) {
        make_tables();
    }
    memset(e, 0, sizeof(*e));
    e->E_frame = -100;
    int i100 = 0;
    for (int i = 1; i < NP; i++) {
        if (fabs(P_GRID[i] - 100) < fabs(P_GRID[i100] - 100)) {
            i100 = i;
        }
    }
    e->hist_i[0] = e->hist_i[1] = i100;
    ambe_parms_init(&e->qcur);
    ambe_parms_init(&e->qprev);
}

/* The input's DC taken out (a microphone's bias, 2500 on the DM-32: with the click at the start of a transmission it
   threw the pitch tracker off to P = 20 and every band went unvoiced; DVSI's encoder high-passes its input too):
   y = x - x1 + (1 - 1/128) y1, about 10 Hz, in integers so that all three encoders agree (py dc_block()) */
#define DC_POLE 32512 /* (1 - 1/128) in Q15 */

static int16_t dc_block(ambe_encoder *e, int16_t x)
{
    e->dc_y = ((int64_t)(x - e->dc_x) << 15) + ((e->dc_y * DC_POLE) >> 15);
    e->dc_x = x;
    int64_t y = e->dc_y >> 15;
    return (int16_t)(y > 32767 ? 32767 : y < -32768 ? -32768 : y);
}

int ambe_enc_frame(ambe_encoder *e, const int16_t in[AMBE_N], uint8_t out9[9])
{
    memmove(e->x, e->x + N, sizeof(double) * (AMBE_ENC_HIST - N));
    if (e->n_in == 0) {
        e->dc_x = in[0]; /* the DC there from the start: no step into the blocker */
    }
    for (int i = 0; i < N; i++) {
        e->x[AMBE_ENC_HIST - N + i] = dc_block(e, in[i]);
    }
    e->n_in += N;
    long k = e->n_in / N - 1 - AMBE_ENC_DELAY;
    if (k < 0) {
        return 0;
    }
    /* voice activity (py vad, peak_level) */
    double edb = frame_energy_db(e, k);
    if (!e->have_level) {
        e->floor_db = e->peak_db = edb;
        e->have_level = 1;
    } else {
        e->floor_db = edb < e->floor_db + FLOOR_RISE ? edb : e->floor_db + FLOOR_RISE;
        e->peak_db = edb > e->peak_db - PEAK_DECAY ? edb : e->peak_db - PEAK_DECAY;
    }
    double thr = e->floor_db + VAD_MARGIN;
    thr = e->peak_db - VAD_BELOW_PEAK > thr ? e->peak_db - VAD_BELOW_PEAK : thr;
    thr = VAD_MIN_DB > thr ? VAD_MIN_DB : thr;
    int raw = edb > thr;
    int speech = raw || e->last_raw;
    e->last_raw = raw;
    double level = edb - e->peak_db;

    double sr[NBINS], si[NBINS];
    spectrum(e, k, sr, si);
    uint8_t d49[49];
    double log2a[64], en[64];
    int voiced[64];
    if (speech) {
        double w0 = pitch(e, k, sr, si);
        int b0 = harmonics(e, sr, si, w0, level, e->track_e, voiced, log2a, en);
        quantize(e, b0, voiced, log2a, en, d49);
    } else {
        noise_amplitudes(sr, si, log2a);
        quantize(e, 124, NULL, log2a, NULL, d49);
    }
    ambe_encode_frame(d49, out9);
    return 1;
}
