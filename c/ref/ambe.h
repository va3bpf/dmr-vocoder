/*
 * REFERENCE copy (doubles, bit-exact with the Python; c/ambe is the embedded version). Our AMBE+2 (DMR, 3600x2450) vocoder in C: the encoder and decoder of py/ambe_encoder.py, py/ambe_analysis.py and
 * py/ambe_decoder.py (which stay the reference: c/test compares the two).
 *
 * Frames are 9 bytes, the 72 channel bits MSB first (as a DMR voice burst carries three of them).
 * Floating point (ambe_real) for now; the radio's CK803S has no FPU, a fixed-point version comes next.
 */
#ifndef AMBE_H
#define AMBE_H

#include <stdint.h>

typedef double ambe_real;

#define AMBE_N 160   /* samples a frame, 20 ms at 8 kHz */

/* ---- frame layer (ambe_frame.c) ---- */

/* 49 vocoder bits (one a byte, 0 / 1) -> 9 bytes */
void ambe_encode_frame(const uint8_t d49[49], uint8_t out9[9]);
/* 9 bytes -> 49 vocoder bits; returns the bit errors corrected in C0 + C1 */
int ambe_decode_frame(const uint8_t in9[9], uint8_t d49[49]);
/* b0..b8 <-> 49 bits */
void ambe_unpack_b(const uint8_t d49[49], int b[9]);
void ambe_pack_b(const int b[9], uint8_t d49[49]);

/* ---- parameters (shared by the decoder and the encoder's decoder mirror) ---- */

typedef struct {
    ambe_real w0;
    int L;
    ambe_real gamma;
    int repeat;
    int Vl[58];
    ambe_real Ml[58];
    ambe_real log2Ml[58];
    ambe_real PHIl[58];
    ambe_real PSIl[58];
} ambe_parms;

void ambe_parms_init(ambe_parms *p);
/* 0 voice / silence, 2 erasure, 3 tone; updates prev as mbelib (py decode_parms, unvc_scale False) */
int ambe_decode_parms(const uint8_t d49[49], ambe_parms *cur, ambe_parms *prev);

/* ---- decoder (ambe_dec.c) ---- */

typedef struct {
    ambe_parms cur, prev, prev_enh;
    ambe_real noise_tail[AMBE_N];
    ambe_real hp[2][4];          /* the two high-pass sections' x1 x2 y1 y2 */
    int ntones;                  /* tone synthesis: frequencies playing, their phase and level */
    ambe_real tone_f[4], tone_phase[4], tone_level[4];
    uint32_t rng;
} ambe_decoder;

void ambe_dec_init(ambe_decoder *d);
/* 9 bytes -> 160 samples; returns the frame kind: 'V' voice, 'R' repeat, 'E' erasure, 'T' tone, 'M' muted */
int ambe_dec_frame(ambe_decoder *d, const uint8_t in9[9], int16_t out[AMBE_N]);

/* ---- encoder (ambe_enc.c) ---- */

#define AMBE_ENC_HIST 1024   /* input history kept (samples) */

typedef struct {
    /* input: x[0..AMBE_ENC_HIST) the latest samples, x_lp the same low-passed to 1 kHz */
    ambe_real x[AMBE_ENC_HIST];
    ambe_real x_lp[AMBE_ENC_HIST];
    long n_in;                   /* samples received */
    int32_t dc_x;                /* the input's DC blocker: the last input sample */
    int64_t dc_y;                /* and the last output, Q15 */
    long k;                      /* the next frame to encode */
    /* pitch tracker */
    ambe_real E[3][217];         /* E(P) of frames k, k + 1, k + 2 */
    long E_frame;                /* the frame E[0] belongs to */
    int hist_i[2];
    ambe_real hist_e[2];
    ambe_real track_e, last_w0, prev_p;
    int have_last;
    int prev_bands[8];
    /* voice activity */
    ambe_real floor_db, peak_db;
    int have_level, last_raw;
    /* quantizer: the decoder mirror */
    ambe_parms qcur, qprev;
} ambe_encoder;

void ambe_enc_init(ambe_encoder *e);
/* 160 input samples in; a frame out (returns 1) once the look-ahead is there: the output lags the input by
   AMBE_ENC_DELAY frames. At the end, push AMBE_ENC_DELAY frames of zeros to get the last ones. */
#define AMBE_ENC_DELAY 1
int ambe_enc_frame(ambe_encoder *e, const int16_t in[AMBE_N], uint8_t out9[9]);

#endif
