/*
 * AMBE+2 decoder: 9 bytes -> 160 samples (py/ambe_decoder.py Decoder with its defaults: mbelib's parameter
 * decoding, full-strength spectral enhancement, voiced harmonics as sinusoids (phase-continuous below the 8th
 * harmonic, a fixed phase offset each), unvoiced bands as shaped noise, silence frames 12 dB down without touching
 * the prediction, tone frames, a 4th-order 120 Hz high-pass).
 * The noise comes from our own generator, so the output is not sample-exact against the Python.
 */
#include <math.h>
#include <string.h>

#include "ambe.h"
#include "ambe_tables.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N AMBE_N
#define GAIN 2.35
#define UV_GAIN 0.77
#define SILENCE_GAIN 0.25
#define INTERP_MAX 8

/* numpy default_rng(12345).uniform(-pi, pi, 58), [0] = 0 (py PHASE_DISP) */
static const double PHASE_DISP[58] = {
    0, -1.1513413075988739, 1.8684022723757732, 1.1074407575842988, -0.6841788717502677, -1.0504610719949992,
    0.61769211610616948, -1.9683071622563955, 1.0854582374791386, 2.7759292717339115, -1.5818188268589761,
    2.8204034578689541, 1.0507839081303949, -2.5390481536760139, -0.36543215499505832, 2.4283249506386113,
    1.2406369293099075, -1.0903031508716263, 1.46981399877098, -1.7584439353097627, -2.6289188530965482,
    -2.136938962232406, -1.0046801685133291, -0.21869786524853163, -1.4676199631095441, 1.9840816583527605,
    -1.9270881868457901, -2.3281144564190939, -2.5656460334963844, 0.6193210952172592, 2.228909121403631,
    0.63850509270728395, 2.7142629235622007, 1.4123429453412095, 2.265410739929778, 2.697608966674812,
    0.29019525366350374, 2.7499803038793873, -0.03149170125543499, -1.4214250158689081, -0.30298331668779266,
    1.0369701386166659, -1.0625436210002133, 2.5349762877002604, -1.5263479726369631, -1.0063882357533678,
    -1.5151687825230518, -0.90825655331515165, -3.110036400170558, 0.80804618192363042, -1.367329775294698,
    -2.7137850833993262, 0.73405811335010807, -2.0337017087298941, -1.2290640113355771, -0.37141912136370347,
    -2.1978435111206545, -1.7723052230410361};

/* dual tones: index 128..163 -> (higher, lower) Hz (py ambe_tone.DUAL) */
static const uint16_t DUAL[36][2] = {
    {1336, 941}, {1209, 697}, {1336, 697}, {1477, 697}, {1209, 770}, {1336, 770}, {1477, 770}, {1209, 852},
    {1336, 852}, {1477, 852}, {1633, 697}, {1633, 770}, {1633, 852}, {1633, 941}, {1209, 941}, {1477, 941},
    {1162, 820}, {1052, 606}, {1162, 606}, {1279, 606}, {1052, 672}, {1162, 672}, {1279, 672}, {1052, 743},
    {1162, 743}, {1279, 743}, {1430, 606}, {1430, 672}, {1430, 743}, {1430, 820}, {1052, 820}, {1279, 820},
    {440, 350}, {480, 440}, {620, 480}, {490, 350}};

/* ---- random numbers ---- */

static double rnd_uniform(ambe_decoder *d)
{
    uint32_t x = d->rng; /* xorshift32 */
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    d->rng = x;
    return (x >> 8) * (1.0 / 16777216.0);
}

static double rnd_phase(ambe_decoder *d)
{
    return rnd_uniform(d) * 2 * M_PI - M_PI;
}

static void rnd_normal2(ambe_decoder *d, double *a, double *b)
{
    double u = rnd_uniform(d), v = rnd_uniform(d);
    double r = sqrt(-2 * log(u > 1e-12 ? u : 1e-12));
    *a = r * cos(2 * M_PI * v);
    *b = r * sin(2 * M_PI * v);
}

/* ---- spectral amplitude enhancement (mbe_spectralAmpEnhance, full strength) ---- */

static void spectral_amp_enhance(ambe_parms *cur)
{
    int L = cur->L;
    ambe_real Rm0 = 0, Rm1 = 0;
    for (int l = 1; l <= L; l++) {
        ambe_real m2 = cur->Ml[l] * cur->Ml[l];
        Rm0 += m2;
        Rm1 += m2 * cos(cur->w0 * l);
    }
    ambe_real R2m0 = Rm0 * Rm0, R2m1 = Rm1 * Rm1;
    ambe_real nw[58], s = 0;
    for (int l = 1; l <= L; l++) {
        ambe_real M = cur->Ml[l];
        nw[l] = M;
        if (M != 0 && 8 * l > L) {
            ambe_real w = sqrt(M) * pow(0.96 * M_PI * ((R2m0 + R2m1) - 2 * Rm0 * Rm1 * cos(cur->w0 * l)) /
                                            (cur->w0 * Rm0 * (R2m0 - R2m1)), 0.25);
            if (w > 1.2) {
                nw[l] = 1.2 * M;
            } else if (w < 0.5) {
                nw[l] = 0.5 * M;
            } else {
                nw[l] = w * M;
            }
        }
        s += nw[l] * nw[l];
    }
    ambe_real g = s == 0 ? 1.0 : sqrt(Rm0 / s);
    for (int l = 1; l <= L; l++) {
        cur->Ml[l] = nw[l] * g;
    }
}

/* ---- voiced synthesis (py synthesize, voiced_only, phase_interp, phase_disp) ---- */

static void synth_voiced(ambe_decoder *d, ambe_parms *cur, ambe_parms *prev, ambe_real *out)
{
    int num_uv = 0;
    for (int l = 1; l <= cur->L; l++) {
        num_uv += cur->Vl[l] == 0;
    }
    ambe_real cw0 = cur->w0, pw0 = prev->w0;
    int maxl;
    if (cur->L > prev->L) {
        maxl = cur->L;
        for (int l = prev->L + 1; l <= maxl; l++) {
            prev->Ml[l] = 0;
            prev->Vl[l] = 1;
        }
    } else {
        maxl = prev->L;
        for (int l = cur->L + 1; l <= maxl; l++) {
            cur->Ml[l] = 0;
            cur->Vl[l] = 1;
        }
    }
    for (int l = 1; l <= 56; l++) {
        cur->PSIl[l] = prev->PSIl[l] + (pw0 + cw0) * (l * N / 2.0);
        if (l <= cur->L / 4) {
            cur->PHIl[l] = cur->PSIl[l];
        } else {
            cur->PHIl[l] = cur->PSIl[l] + num_uv * rnd_phase(d) / cur->L;
        }
        cur->PHIl[l] += PHASE_DISP[l];
    }
    for (int n = 0; n < N; n++) {
        out[n] = 0;
    }
    for (int l = 1; l <= maxl; l++) {
        if (prev->Vl[l] && cur->Vl[l] && l < INTERP_MAX && fabs(cw0 - pw0) < 0.1 * cw0) {
            ambe_real dphi = cur->PHIl[l] - prev->PHIl[l] - (pw0 + cw0) * l * N / 2;
            ambe_real dw = (dphi - 2 * M_PI * floor((dphi + M_PI) / (2 * M_PI))) / N;
            for (int n = 0; n < N; n++) {
                ambe_real theta = prev->PHIl[l] + (pw0 * l + dw) * n + (cw0 - pw0) * l * n * n / (2.0 * N);
                ambe_real amp = prev->Ml[l] + (ambe_real)n / N * (cur->Ml[l] - prev->Ml[l]);
                out[n] += amp * cos(theta);
            }
            continue;
        }
        if (prev->Vl[l]) {
            for (int n = 0; n < N; n++) {
                out[n] += Ws[N + n] * prev->Ml[l] * cos(pw0 * l * n + prev->PHIl[l]);
            }
        }
        if (cur->Vl[l]) {
            for (int n = 0; n < N; n++) {
                out[n] += Ws[n] * cur->Ml[l] * cos(cw0 * l * (n - N) + cur->PHIl[l]);
            }
        }
    }
}

/* ---- unvoiced synthesis (py NoiseSynth): white noise kept in the unvoiced bands, each with the power of a
   sinusoid of amplitude Ml, 2N samples a frame, cross-faded with sqrt(Ws) ---- */

static void synth_noise(ambe_decoder *d, const ambe_parms *p, ambe_real *out)
{
    enum { M = 2 * N, K = M / 2 + 1 };
    ambe_real xr[K], xi[K], sr[K], si[K];
    for (int k = 0; k < K; k++) {
        rnd_normal2(d, &xr[k], &xi[k]);
        sr[k] = si[k] = 0;
    }
    int any = 0;
    for (int l = 1; l <= p->L; l++) {
        if (p->Vl[l] || p->Ml[l] == 0) {
            continue;
        }
        ambe_real lo = (l - 0.5) * p->w0, hi = (l + 0.5) * p->w0, e = 0;
        int k0 = -1, k1 = -1;
        for (int k = 1; k < M / 2; k++) {
            ambe_real wk = k * 2 * M_PI / M;
            if (wk >= lo && wk < hi) {
                if (k0 < 0) {
                    k0 = k;
                }
                k1 = k;
                e += xr[k] * xr[k] + xi[k] * xi[k];
            }
        }
        if (k0 < 0 || e == 0) {
            continue;
        }
        ambe_real g = sqrt((p->Ml[l] * p->Ml[l] / 2) * M * M / 2 / e) * UV_GAIN;
        for (int k = k0; k <= k1; k++) {
            sr[k] = xr[k] * g;
            si[k] = xi[k] * g;
        }
        any = 1;
    }
    ambe_real u[M];
    for (int n = 0; n < M; n++) {
        u[n] = 0;
    }
    if (any) {
        /* irfft: u[n] = 2 / M sum_k Re(S_k e^(i 2 pi k n / M)), S_0 = S_(M/2) = 0 here */
        static ambe_real cs[M], sn[M];
        static int ready;
        if (!ready) {
            for (int m = 0; m < M; m++) {
                cs[m] = cos(2 * M_PI * m / M);
                sn[m] = sin(2 * M_PI * m / M);
            }
            ready = 1;
        }
        for (int k = 1; k < M / 2; k++) {
            if (sr[k] == 0 && si[k] == 0) {
                continue;
            }
            int idx = 0;
            for (int n = 0; n < M; n++) {
                u[n] += sr[k] * cs[idx] - si[k] * sn[idx];
                idx += k;
                if (idx >= M) {
                    idx -= M;
                }
            }
        }
        for (int n = 0; n < M; n++) {
            u[n] *= 2.0 / M;
        }
    }
    for (int n = 0; n < N; n++) {
        out[n] += sqrt(Ws[N + n]) * d->noise_tail[n] + sqrt(Ws[n]) * u[n];
        d->noise_tail[n] = u[N + n];
    }
}

/* ---- tones (py ambe_tone) ---- */

static void tone_frame(ambe_decoder *d, const ambe_real *freqs, int nf, ambe_real a, ambe_real *out)
{
    ambe_real f[8], a0[8], ph[8];
    int n = 0;
    for (int i = 0; i < nf; i++) { /* the new ones, then the old ones that stop */
        f[n] = freqs[i];
        a0[n] = 0;
        ph[n] = 0;
        for (int j = 0; j < d->ntones; j++) {
            if (d->tone_f[j] == freqs[i]) {
                a0[n] = d->tone_level[j];
                ph[n] = d->tone_phase[j];
            }
        }
        n++;
    }
    int nnew = n;
    for (int j = 0; j < d->ntones; j++) {
        int found = 0;
        for (int i = 0; i < nf; i++) {
            found |= d->tone_f[j] == freqs[i];
        }
        if (!found) {
            f[n] = d->tone_f[j];
            a0[n] = d->tone_level[j];
            ph[n] = d->tone_phase[j];
            n++;
        }
    }
    d->ntones = 0;
    for (int i = 0; i < n; i++) {
        ambe_real a1 = i < nnew ? a : 0;
        ambe_real w = 2 * M_PI * f[i] / 8000;
        for (int k = 0; k < N; k++) {
            out[k] += (a0[i] + (a1 - a0[i]) * k / N) * cos(ph[i] + w * k);
        }
        if (a1 != 0 && d->ntones < 4) {
            d->tone_f[d->ntones] = f[i];
            d->tone_phase[d->ntones] = fmod(ph[i] + w * N, 2 * M_PI);
            d->tone_level[d->ntones] = a1;
            d->ntones++;
        }
    }
}

static void tone_off(ambe_decoder *d, ambe_real *out)
{
    if (d->ntones) {
        tone_frame(d, NULL, 0, 0, out);
    }
}

/* tone frame: the index (majority of 4 copies) and the amplitude -> frequencies; -1 if not a tone */
static int tone_parse(const uint8_t d49[49], ambe_real *freqs, ambe_real *peak)
{
    int amp = 0, c[4] = {0};
    for (int i = 6; i < 12; i++) {
        amp = (amp << 1) | d49[i];
    }
    amp = (amp << 1) | d49[44];
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 8; i++) {
            c[k] = (c[k] << 1) | d49[12 + 8 * k + i];
        }
    }
    int idx = c[0], best = 0;
    for (int k = 0; k < 4; k++) { /* max(copies, key=copies.count): the first of the most frequent */
        int cnt = 0;
        for (int j = 0; j < 4; j++) {
            cnt += c[j] == c[k];
        }
        if (cnt > best) {
            best = cnt;
            idx = c[k];
        }
    }
    *peak = 32767 * pow(10, 0.711 * (amp - 127) / 20);
    if (idx == 255) {
        return 0;
    }
    if (idx >= 5 && idx <= 122) {
        freqs[0] = 31.25 * idx;
        return 1;
    }
    if (idx >= 128 && idx <= 163) {
        freqs[0] = DUAL[idx - 128][0];
        freqs[1] = DUAL[idx - 128][1];
        return 2;
    }
    return -1;
}

/* ---- high-pass: 4th-order Butterworth at 120 Hz, two RBJ sections ---- */

static void highpass(ambe_decoder *d, ambe_real *x)
{
    static const double Q[2] = {0.5412, 1.3066};
    for (int s = 0; s < 2; s++) {
        double w = 2 * M_PI * 120 / 8000, alpha = sin(w) / (2 * Q[s]), c = cos(w), a0 = 1 + alpha;
        double b0 = (1 + c) / 2 / a0, b1 = -(1 + c) / a0, b2 = b0, a1 = -2 * c / a0, a2 = (1 - alpha) / a0;
        ambe_real *st = d->hp[s];
        for (int n = 0; n < N; n++) {
            ambe_real v = x[n];
            ambe_real o = b0 * v + b1 * st[0] + b2 * st[1] - a1 * st[2] - a2 * st[3];
            st[1] = st[0];
            st[0] = v;
            st[3] = st[2];
            st[2] = o;
            x[n] = o;
        }
    }
}

static void reset_voice(ambe_decoder *d)
{
    ambe_parms_init(&d->cur);
    ambe_parms_init(&d->prev);
    ambe_parms_init(&d->prev_enh);
    memset(d->noise_tail, 0, sizeof(d->noise_tail));
}

void ambe_dec_init(ambe_decoder *d)
{
    memset(d, 0, sizeof(*d));
    d->rng = 0x12345678;
    reset_voice(d);
}

int ambe_dec_frame(ambe_decoder *d, const uint8_t in9[9], int16_t pcm[AMBE_N])
{
    uint8_t d49[49];
    int errs = ambe_decode_frame(in9, d49);
    ambe_real out[N];
    int kind;
    for (int n = 0; n < N; n++) {
        out[n] = 0;
    }
    ambe_real freqs[2], peak;
    int nf = -1;
    if (d49[0] & d49[1] & d49[2] & d49[3] & d49[4] & d49[5]) {
        nf = tone_parse(d49, freqs, &peak);
    }
    if (nf >= 0 && errs <= 3) {
        reset_voice(d);
        tone_frame(d, freqs, nf, peak, out);
        kind = 'T';
    } else {
        ambe_parms *cur = &d->cur, *prev = &d->prev;
        int b0 = (d49[0] << 6) | (d49[1] << 5) | (d49[2] << 4) | (d49[3] << 3) | (d49[37] << 2) | (d49[38] << 1) |
                 d49[39];
        int silence = b0 == 124 || b0 == 125;
        static ambe_parms keep;
        if (silence) {
            keep = *prev;
        }
        int bad = ambe_decode_parms(d49, cur, prev);
        if (bad == 2 || bad == 3) {
            cur->repeat = 0;
        } else if (errs > 3) {
            *cur = *prev;
            cur->repeat++;
        } else {
            cur->repeat = 0;
        }
        if (bad == 0 && cur->repeat <= 3) {
            *prev = silence ? keep : *cur;
            if (silence) {
                for (int l = 0; l < 58; l++) {
                    cur->Ml[l] *= SILENCE_GAIN;
                }
            }
            spectral_amp_enhance(cur);
            synth_voiced(d, cur, &d->prev_enh, out);
            synth_noise(d, cur, out);
            d->prev_enh = *cur;
            for (int n = 0; n < N; n++) {
                out[n] *= GAIN;
            }
            tone_off(d, out);
            kind = cur->repeat ? 'R' : 'V';
        } else {
            reset_voice(d);
            tone_off(d, out);
            kind = bad == 2 ? 'E' : bad == 3 ? 'T' : 'M';
        }
    }
    highpass(d, out);
    for (int n = 0; n < N; n++) {
        ambe_real v = out[n];
        pcm[n] = (int16_t)(v > 32760 ? 32760 : v < -32760 ? -32760 : v);
    }
    return kind;
}
