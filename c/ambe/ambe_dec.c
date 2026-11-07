/*
 * AMBE+2 decoder: 9 bytes -> 160 samples (py/ambe_decoder.py Decoder with its defaults: mbelib's parameter
 * decoding, full-strength spectral enhancement, voiced harmonics as sinusoids (phase-continuous below the 8th
 * harmonic, a fixed phase offset each), unvoiced bands as shaped noise, silence frames 12 dB down without touching
 * the prediction, tone frames, a 4th-order 120 Hz high-pass).
 *
 * The synthesis is integer: phases in uint32 (2^32 = 2 pi), the cosine from a Q12 table, the output in Q4 until
 * the high-pass (Q8); the amplitudes Ml in Q6, the enhancement and the noise gains in log2 (Q12). Only tone frames
 * use float. The noise is our own, so the output is not sample-exact against c/ref.
 */
#include <string.h>

#include "ambe.h"
#include "ambe_fx.h"

#define N AMBE_N
/* gains (py Decoder): output 2.35, unvoiced 0.77, silence frames 0.25; in amp_q4() and synth_noise() */
#define INTERP_MAX 8
#define NM 512                   /* the noise's IFFT */
#define AMAX ((1 << 19) - 1)     /* largest amplitude, Q4 (32768 PCM): x Q12 cosine stays in 32 bits */

/* dual tones: index 128..163 -> (higher, lower) Hz (py ambe_tone.DUAL) */
static const uint16_t DUAL[36][2] = {
    {1336, 941}, {1209, 697}, {1336, 697}, {1477, 697}, {1209, 770}, {1336, 770}, {1477, 770}, {1209, 852},
    {1336, 852}, {1477, 852}, {1633, 697}, {1633, 770}, {1633, 852}, {1633, 941}, {1209, 941}, {1477, 941},
    {1162, 820}, {1052, 606}, {1162, 606}, {1279, 606}, {1052, 672}, {1162, 672}, {1279, 672}, {1052, 743},
    {1162, 743}, {1279, 743}, {1430, 606}, {1430, 672}, {1430, 743}, {1430, 820}, {1052, 820}, {1279, 820},
    {440, 350}, {480, 440}, {620, 480}, {490, 350}};

/* the high-pass: 4th-order Butterworth at 120 Hz as two RBJ sections (Q 0.5412, 1.3066): b0 b1 b2 a1 a2, Q14 */
static const int16_t HP[2][5] = {{15040, -30080, 15040, -30013, 13763}, {15779, -31559, 15779, -31489, 15245}};

/* profiling on the radio (-DAMBE_PROFILE): ambe_prof[13] the parameters, [14] the voiced, [15] the unvoiced synthesis */
#ifdef AMBE_PROFILE
#define PROF(i, x)                                                                                                         do {                                                                                                                       uint32_t t_ = ambe_now();                                                                                              x;                                                                                                                     ambe_prof[i] += ambe_now() - t_;                                                                                   } while (0)
#else
#define PROF(i, x) x
#endif

static uint32_t rnd(ambe_decoder *d)
{
    uint32_t x = d->rng; /* xorshift32 */
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    d->rng = x;
    return x;
}

/* about normal, sd 1180: the sum of four uniform numbers */
static int32_t rnd_normal(ambe_decoder *d)
{
    int32_t s = 0;
    for (int i = 0; i < 4; i++) {
        s += (int32_t)(rnd(d) >> 16) - 32768;
    }
    return s >> 5;
}

static int32_t clamp_amp(float a)
{
    return a > AMAX ? AMAX : a < -AMAX ? -AMAX : (int32_t)a;
}

/* (a * b) >> 12 for a 32-bit a and a Q12 b */
static inline int32_t mulq12(int32_t a, int32_t b)
{
    return (a >> 12) * b + (((a & 0xFFF) * b) >> 12);
}

static inline int32_t mulq14(int32_t a, int32_t b)
{
    return (a >> 14) * b + (((a & 0x3FFF) * b) >> 14);
}

/* Ml (Q6) -> the output amplitude GAIN x Ml, Q4 */
static int32_t amp_q4(int32_t ml)
{
    int32_t a = mulq12(ml, 2406); /* 2.35 / 4 */
    return a > AMAX ? AMAX : a;
}

/* ---- spectral amplitude enhancement (mbe_spectralAmpEnhance, full strength), integer ----
   W_l^4 = M_l^2 0.96 pi (1 + rho^2 - 2 rho cos(w0 l)) / (w0 Rm0 (1 - rho^2)), rho = Rm1 / Rm0: mbelib's formula
   divided through by Rm0^3, evaluated in log2 (Q12) */

static void spectral_amp_enhance(ambe_parms *cur)
{
    int L = cur->L;
    int32_t mx = 0;
    for (int l = 1; l <= L; l++) {
        mx = cur->Ml[l] > mx ? cur->Ml[l] : mx;
    }
    if (mx == 0) {
        return;
    }
    int sh = (31 - __builtin_clz((uint32_t)mx)) - 11; /* m_l = Ml >> sh < 2^12: the sums fit 32 bits */
    int32_t m[58];
    uint32_t Rm0 = 0;
    int32_t Rm1 = 0;
    for (int l = 1; l <= L; l++) {
        m[l] = sh >= 0 ? cur->Ml[l] >> sh : cur->Ml[l] << -sh;
        int32_t m2 = m[l] * m[l];
        Rm0 += (uint32_t)m2;
        Rm1 += (m2 >> 12) * cos_q12(cur->w0q * (uint32_t)l);
    }
    if (Rm0 == 0) {
        return;
    }
    int32_t rho = (int32_t)(((int64_t)Rm1 << 15) / (int64_t)Rm0); /* Q15 */
    int32_t rho2 = (rho * rho) >> 15, one_m = 32768 - rho2;
    if (one_m <= 0) {
        return;
    }
    /* log2 of the factors common to every l: 0.96 pi / w0 = 0.48 x 2^32 / w0q; / Rm0; / (1 - rho^2) */
    int32_t common = 126735 - log2_q12(cur->w0q) - log2_q12(Rm0) - (log2_q12((uint32_t)one_m) - (15 << 12));
    uint32_t s = 0;
    int32_t f[58];
    for (int l = 1; l <= L; l++) {
        f[l] = 1 << 14;
        if (m[l] != 0 && 8 * l > L) {
            int32_t c = cos_q12(cur->w0q * (uint32_t)l);
            int32_t a = 32768 + rho2 - ((2 * rho * c) >> 12); /* 1 + rho^2 - 2 rho cos, Q15 */
            int32_t lw = -4096;                              /* log2 0.5 */
            if (a > 0) {
                lw = (log2_q12((uint32_t)(m[l] * m[l])) + log2_q12((uint32_t)a) - (15 << 12) + common) >> 2;
                lw = lw > 1077 ? 1077 : lw < -4096 ? -4096 : lw; /* 0.5 .. 1.2 */
            }
            f[l] = (int32_t)exp2_q12(lw, 14);
        }
        int32_t nm = (m[l] * f[l]) >> 14;
        s += (uint32_t)(nm * nm);
    }
    /* energy kept: g = sqrt(Rm0 / s) */
    int32_t g = s ? (int32_t)exp2_q12((log2_q12(Rm0) - log2_q12(s)) / 2, 14) : (1 << 14);
    for (int l = 1; l <= L; l++) {
        int32_t k = (f[l] * g) >> 14;
        cur->Ml[l] = mulq14(cur->Ml[l], k > 32767 ? 32767 : k);
    }
}

/* ---- voiced synthesis (py synthesize, voiced_only, phase_interp, phase_disp), into out (Q4) ---- */

static void synth_voiced(ambe_decoder *d, ambe_parms *cur, ambe_parms *prev, int32_t *out)
{
    int num_uv = 0;
    for (int l = 1; l <= cur->L; l++) {
        num_uv += cur->Vl[l] == 0;
    }
    uint32_t cw0 = cur->w0q, pw0 = prev->w0q;
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
        cur->PSIl[l] = prev->PSIl[l] + (pw0 + cw0) * (uint32_t)(l * N / 2);
        cur->PHIl[l] = cur->PSIl[l];
        if (l > cur->L / 4) { /* + num_uv x a random phase / L */
            cur->PHIl[l] += (uint32_t)((((int32_t)rnd(d) >> 6) * num_uv / cur->L) << 6);
        }
        cur->PHIl[l] += (uint32_t)AmbePhaseDispQ32[l];
    }
    int32_t *acc_p = ambe_scratch, *acc_c = ambe_scratch + N;
    memset(acc_p, 0, sizeof(int32_t) * 2 * N);
    int dw0_small = (int32_t)(cw0 - pw0) < (int32_t)(cw0 / 10) && (int32_t)(pw0 - cw0) < (int32_t)(cw0 / 10);
    for (int l = 1; l <= maxl; l++) {
        int32_t ap = amp_q4(prev->Ml[l]), ac = amp_q4(cur->Ml[l]);
        if (prev->Vl[l] && cur->Vl[l] && l < INTERP_MAX && dw0_small) {
            /* one sinusoid: amplitude ramp, the phase lands on PHIl at n = N (IMBE eq. 137, 138) */
            int32_t dphi = (int32_t)(cur->PHIl[l] - prev->PHIl[l] - (pw0 + cw0) * (uint32_t)(l * N / 2));
            int32_t c = (int32_t)(cw0 - pw0) * l / (2 * N); /* the phase's second difference / 2 */
            uint32_t th = prev->PHIl[l], inc = pw0 * (uint32_t)l + (uint32_t)(dphi / N) + (uint32_t)c;
            int32_t amp = ap << 8, damp = ((ac - ap) << 8) / N;
            for (int n = 0; n < N; n++) {
                out[n] += ((amp >> 8) * cos_q12(th)) >> 12;
                th += inc;
                inc += (uint32_t)(2 * c);
                amp += damp;
            }
            continue;
        }
        if (prev->Vl[l] && ap) {
            uint32_t th = prev->PHIl[l], step = pw0 * (uint32_t)l;
            for (int n = 0; n < N; n++) {
                acc_p[n] += (ap * cos_q12(th)) >> 12;
                th += step;
            }
        }
        if (cur->Vl[l] && ac) {
            uint32_t step = cw0 * (uint32_t)l, th = cur->PHIl[l] - step * N;
            for (int n = 0; n < N; n++) {
                acc_c[n] += (ac * cos_q12(th)) >> 12;
                th += step;
            }
        }
    }
    for (int n = 0; n < N; n++) {
        out[n] += mul32x16(acc_p[n], AmbeWsQ15[N + n]) + mul32x16(acc_c[n], AmbeWsQ15[n]);
    }
}

/* ---- unvoiced synthesis (py NoiseSynth): white noise kept in the unvoiced bands, each with the power of a
   sinusoid of amplitude Ml, 2N samples a frame (of an NM-point IFFT), cross-faded with sqrt(Ws) ---- */

/* in-place inverse FFT (e^+i), no scaling, Q12 twiddles from the cosine table */
static void ifft(int32_t *re, int32_t *im, int n)
{
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            int32_t t = re[i];
            re[i] = re[j];
            re[j] = t;
            t = im[i];
            im[i] = im[j];
            im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        uint32_t step = 0xFFFFFFFFu / (uint32_t)len + 1; /* 2^32 / len */
        for (int j = 0; j < len / 2; j++) {
            uint32_t ph = step * (uint32_t)j;
            int32_t wr = cos_q12(ph), wi = cos_q12(ph - (1u << 30)); /* e^(+i 2 pi j / len) */
            for (int i = j; i < n; i += len) {
                int32_t *ar = &re[i], *ai = &im[i], *br = &re[i + len / 2], *bi = &im[i + len / 2];
                int32_t vr = mulq12(*br, wr) - mulq12(*bi, wi);
                int32_t vi = mulq12(*br, wi) + mulq12(*bi, wr);
                *br = *ar - vr;
                *bi = *ai - vi;
                *ar += vr;
                *ai += vi;
            }
        }
    }
}

static void synth_noise(ambe_decoder *d, const ambe_parms *p, int32_t *out)
{
    int32_t *xr = ambe_scratch, *xi = ambe_scratch + NM; /* after synth_voiced */
    uint8_t in[NM / 2];
    memset(xr, 0, sizeof(int32_t) * 2 * NM);
    memset(in, 0, sizeof(in));
    int any = 0;
    uint32_t wb = p->w0q >> 7; /* w0 in NM-point bins, Q16 */
    for (int k = 1; k < NM / 2; k++) {
        xr[k] = rnd_normal(d);
        xi[k] = rnd_normal(d);
    }
    for (int l = 1; l <= p->L; l++) {
        if (p->Vl[l] || p->Ml[l] == 0) {
            continue;
        }
        /* bins with (l - 1/2) w0 <= w_k < (l + 1/2) w0, 0 < k < NM / 2 */
        uint32_t lo = (uint32_t)(2 * l - 1) * wb / 2, hi = (uint32_t)(2 * l + 1) * wb / 2; /* Q16 */
        int k0 = (int)((lo + 0xFFFF) >> 16);                                               /* k >= lo */
        if (k0 < 1) {
            k0 = 1;
        }
        int k1 = (int)((hi - 1) >> 16); /* k < hi */
        if (k1 > NM / 2 - 1) {
            k1 = NM / 2 - 1;
        }
        uint32_t e = 0;
        for (int k = k0; k <= k1; k++) {
            e += (uint32_t)(xr[k] * xr[k] + xi[k] * xi[k]);
        }
        if (e == 0) {
            continue;
        }
        /* u = 2 / NM sum Re(S_k e^(...)), S_k = X_k Ml NM / (2 sqrt(e)): the scale per X_k is
           UV_GAIN GAIN Ml / sqrt(e) in Q4, Ml in Q6: log2(0.77 x 2.35 / 4) = -1.1446 */
        int32_t lg = log2_q12((uint32_t)p->Ml[l]) - 4688 - log2_q12(e) / 2;
        uint32_t g = exp2_q12(lg, 8);
        g = g > (1u << 18) ? (1u << 18) : g;
        for (int k = k0; k <= k1; k++) {
            xr[k] = (xr[k] * (int32_t)g) >> 8;
            xi[k] = (xi[k] * (int32_t)g) >> 8;
            in[k] = 1;
        }
        any = 1;
    }
    for (int k = 1; k < NM / 2; k++) { /* bins outside the unvoiced bands carry nothing */
        if (!in[k]) {
            xr[k] = xi[k] = 0;
        }
    }
    if (any) {
        ifft(xr, xi, NM);
    } else {
        memset(xr, 0, sizeof(int32_t) * NM);
    }
    for (int n = 0; n < N; n++) {
        out[n] += mul32x16(d->noise_tail[n], AmbeSqrtWsQ15[N + n]) + mul32x16(xr[n], AmbeSqrtWsQ15[n]);
        d->noise_tail[n] = xr[N + n];
    }
}

/* ---- tones (py ambe_tone) ---- */

static void tone_frame(ambe_decoder *d, const float *freqs, int nf, float a, int32_t *out)
{
    uint32_t w[8], ph[8];
    float a0[8];
    int n = 0;
    for (int i = 0; i < nf; i++) { /* the new ones, then the old ones that stop */
        w[n] = phase_of(freqs[i] / 8000);
        a0[n] = 0;
        ph[n] = 0;
        for (int j = 0; j < d->ntones; j++) {
            if (d->tone_w[j] == w[n]) {
                a0[n] = d->tone_level[j];
                ph[n] = d->tone_phase[j];
            }
        }
        n++;
    }
    int nnew = n;
    for (int j = 0; j < d->ntones; j++) {
        int found = 0;
        for (int i = 0; i < nnew; i++) {
            found |= d->tone_w[j] == w[i];
        }
        if (!found) {
            w[n] = d->tone_w[j];
            a0[n] = d->tone_level[j];
            ph[n] = d->tone_phase[j];
            n++;
        }
    }
    d->ntones = 0;
    for (int i = 0; i < n; i++) {
        float a1 = i < nnew ? a : 0;
        int32_t amp = clamp_amp(a0[i] * 16) << 8, damp = ((clamp_amp(a1 * 16) << 8) - amp) / N;
        uint32_t th = ph[i];
        for (int k = 0; k < N; k++) {
            out[k] += ((amp >> 8) * cos_q12(th)) >> 12;
            th += w[i];
            amp += damp;
        }
        if (a1 != 0 && d->ntones < 4) {
            d->tone_w[d->ntones] = w[i];
            d->tone_phase[d->ntones] = ph[i] + w[i] * N;
            d->tone_level[d->ntones] = a1;
            d->ntones++;
        }
    }
}

static void tone_off(ambe_decoder *d, int32_t *out)
{
    if (d->ntones) {
        tone_frame(d, NULL, 0, 0, out);
    }
}

/* tone frame: the index (majority of 4 copies) and the amplitude -> frequencies; -1 if not a tone */
static int tone_parse(const uint8_t d49[49], float *freqs, float *peak)
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
    *peak = 32767 * exp2_f(0.11811298f * (amp - 127)); /* 0.711 dB a step */
    if (idx == 255) {
        return 0;
    }
    if (idx >= 5 && idx <= 122) {
        freqs[0] = 31.25f * idx;
        return 1;
    }
    if (idx >= 128 && idx <= 163) {
        freqs[0] = DUAL[idx - 128][0];
        freqs[1] = DUAL[idx - 128][1];
        return 2;
    }
    return -1;
}

/* ---- high-pass, Q8 ---- */

static void highpass(ambe_decoder *d, int32_t *x)
{
    for (int s = 0; s < 2; s++) {
        const int16_t *c = HP[s];
        int32_t *st = d->hp[s];
        for (int n = 0; n < N; n++) {
            int32_t v = x[n];
            int32_t o = mulq14(v, c[0]) + mulq14(st[0], c[1]) + mulq14(st[1], c[2]) - mulq14(st[2], c[3]) -
                        mulq14(st[3], c[4]);
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
    int32_t out[N];
    int kind;
    memset(out, 0, sizeof(out));
    float freqs[2], peak;
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
        if (silence) {
            d->keep = *prev;
        }
        int bad;
        PROF(13, bad = ambe_decode_parms(d49, cur, prev));
        if (bad == 2 || bad == 3) {
            cur->repeat = 0;
        } else if (errs > 3) {
            *cur = *prev;
            cur->repeat++;
        } else {
            cur->repeat = 0;
        }
        if (bad == 0 && cur->repeat <= 3) {
            *prev = silence ? d->keep : *cur;
            if (silence) {
                for (int l = 0; l < 58; l++) {
                    cur->Ml[l] >>= 2; /* SILENCE_GAIN 0.25 */
                }
            }
            PROF(13, spectral_amp_enhance(cur));
            PROF(14, synth_voiced(d, cur, &d->prev_enh, out));
            PROF(15, synth_noise(d, cur, out));
            d->prev_enh = *cur;
            tone_off(d, out);
            kind = cur->repeat ? 'R' : 'V';
        } else {
            reset_voice(d);
            tone_off(d, out);
            kind = bad == 2 ? 'E' : bad == 3 ? 'T' : 'M';
        }
    }
    for (int n = 0; n < N; n++) {
        out[n] <<= 4; /* Q4 -> Q8 */
    }
    highpass(d, out);
    for (int n = 0; n < N; n++) {
        int32_t v = (out[n] + 128) >> 8;
        pcm[n] = (int16_t)(v > 32760 ? 32760 : v < -32760 ? -32760 : v);
    }
    return kind;
}
