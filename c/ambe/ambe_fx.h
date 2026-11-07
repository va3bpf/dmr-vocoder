/* Small helpers of the embedded vocoder: table cosine, fast exp2 / log2, 32 x 16 bit products. */
#ifndef AMBE_FX_H
#define AMBE_FX_H

#include <stdint.h>

#include "ambe_tables.h"

/* Hot functions of the encoder: the firmware puts them into RAM (-DAMBE_RAM=...) */
#ifndef AMBE_RAM
#define AMBE_RAM
#endif

/* One work area for the steps of a frame that never run at the same time (the encoder's FFT, E(P), tracker and
   quantizer, the decoder's synthesis): RAM is short on the radio. */
#define AMBE_SCRATCH_WORDS 2052
extern int32_t ambe_scratch[AMBE_SCRATCH_WORDS];

/* cos of a phase (2^32 = 2 pi), Q12: from the quarter table, cos(pi - x) = -cos(x), cos(2 pi - x) = cos(x) */
static inline int32_t cos_q12(uint32_t ph)
{
    uint32_t i = ph >> 20, q = i >> 10, j = i & 1023;
    int32_t v = AmbeCosQ12[(q & 1) ? 1024 - j : j];
    return ((q + 1) & 2) ? -v : v;
}

static inline float cos_f(uint32_t ph)
{
    return cos_q12(ph) * (1.0f / 4096);
}

/* cos(2 pi k / 2048), Q15, any k (modulo 2048): the encoder's FFT twiddles, from their quarter table. Its first two
   entries are 32768 cut to 32767; near -1 the full table had -32768 */
static inline int32_t cos2048_q15(uint32_t k)
{
    uint32_t q = (k >> 9) & 3, j = k & 511, idx = (q & 1) ? 512 - j : j;
    int32_t v = AmbeCos2048Q15[idx];
    return ((q + 1) & 2) ? (idx <= 1 ? -32768 : -v) : v;
}

/* (a * b) >> 15 for a 32-bit a and a Q15 b, without a 64-bit product */
static inline int32_t mul32x16(int32_t a, int32_t b)
{
    return (a >> 15) * b + (((a & 0x7FFF) * b) >> 15);
}

/* 2^x and log2 x for floats, about 1e-5 relative (no libm) */
static inline float exp2_f(float x)
{
    if (x < -126) {
        return 0;
    }
    if (x > 127) {
        x = 127;
    }
    int i = (int)x;
    if (x < i) {
        i--;
    }
    float f = x - i;
    /* 2^f on [0, 1) */
    float p = 1.0f + f * (0.693147f + f * (0.240227f + f * (0.0555036f + f * (0.00961813f + f * 0.00133336f))));
    union {
        float f;
        uint32_t u;
    } v = {p};
    v.u += (uint32_t)i << 23;
    return v.f;
}

static inline float log2_f(float x)
{
    union {
        float f;
        uint32_t u;
    } v = {x};
    if (x <= 0) {
        return -127;
    }
    int e = (int)((v.u >> 23) & 0xFF) - 127;
    v.u = (v.u & 0x7FFFFF) | 0x3F800000; /* mantissa in [1, 2) */
    float m = v.f, t = (m - 1) / (m + 1), t2 = t * t;
    /* log2 m = 2 / ln 2 (t + t^3 / 3 + t^5 / 5 + t^7 / 7), t <= 1/3 */
    return e + 2.88539008f * t * (1 + t2 * (0.333333333f + t2 * (0.2f + t2 * 0.142857143f)));
}

/* log2 x, Q12, for x > 0 (x = 0 gives -32 << 12) */
static inline int32_t log2_q12(uint32_t x)
{
    if (x == 0) {
        return -(32 << 12);
    }
    int e = 31 - __builtin_clz(x);
    uint32_t m = x << (31 - e); /* 1.31 */
    uint32_t i = (m >> 23) & 0xFF, f = (m >> 15) & 0xFF;
    int32_t a = AmbeLog2Q12[i], b = AmbeLog2Q12[i + 1];
    return (e << 12) + a + (((b - a) * (int32_t)f) >> 8);
}

/* 2^(v / 4096) x 2^frac, saturated to 32 bits */
static inline uint32_t exp2_q12(int32_t v, int frac)
{
    int32_t i = v >> 12; /* floor */
    uint32_t f = (uint32_t)v & 0xFFF;
    uint32_t a = AmbeExp2Q15[f >> 4], b = AmbeExp2Q15[(f >> 4) + 1];
    uint32_t m = a + (((b - a) * (f & 15)) >> 4); /* 2^(f / 4096), Q15 */
    int sh = i + frac - 15;
    if (sh >= 16) {
        return 0xFFFFFFFFu;
    }
    if (sh >= 0) {
        return m << sh;
    }
    return sh <= -32 ? 0 : m >> -sh;
}

static inline uint32_t phase_of(float turns)
{
    return (uint32_t)(int64_t)(turns * 4294967296.0f);
}

#endif
