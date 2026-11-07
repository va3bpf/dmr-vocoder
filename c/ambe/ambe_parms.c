/*
 * AMBE+2 parameter decoding, 49 bits -> w0, L, voicing and log2 spectral amplitudes (py/ambe_decoder.py
 * decode_parms, after mbelib's mbe_decodeAmbe2450Parms). The encoder runs it too, as its mirror of the decoder.
 * Integer: the codebooks, gamma and log2 Ml in Q12, Ml linear in Q6.
 */
#include <string.h>

#include "ambe.h"
#include "ambe_fx.h"

void ambe_parms_init(ambe_parms *p)
{
    memset(p, 0, sizeof(*p));
    p->w0 = 0.09378f;
    p->w0q = 64107620; /* 0.09378 / 2 pi x 2^32 */
    p->L = 30;
    for (int l = 0; l < 58; l++) {
        p->PSIl[l] = 1u << 30; /* pi / 2 */
    }
}

static const short *hoc_row(int i, int b)
{
    switch (i) {
    case 1:
        return AmbeHOCb5Q12[b];
    case 2:
        return AmbeHOCb6Q12[b];
    case 3:
        return AmbeHOCb7Q12[b];
    default:
        return AmbeHOCb8Q12[b];
    }
}

int ambe_decode_parms(const uint8_t d49[49], ambe_parms *cur, ambe_parms *prev)
{
    int b[9];
    ambe_unpack_b(d49, b);
    cur->repeat = prev->repeat;
    int b0 = b[0], L;
    int silence = 0;
    if (b0 >= 120 && b0 <= 123) {
        return 2;
    }
    if (b0 == 124 || b0 == 125) {
        silence = 1;
        cur->w0 = 6.2831853f / 32;
        cur->w0q = 1u << 27;
        L = 14;
        cur->L = 14;
        for (int l = 1; l <= L; l++) {
            cur->Vl[l] = 0;
        }
    } else if (b0 >= 126) {
        return 3;
    } else {
        cur->w0 = AmbeW0table[b0] * 6.2831853f;
        cur->w0q = phase_of(AmbeW0table[b0]);
        L = AmbeLtable[b0];
        cur->L = L;
    }
    if (!silence) {
        for (int l = 1; l <= L; l++) {
            cur->Vl[l] = AmbeVuv[b[1]][((cur->w0q >> 4) * (uint32_t)l) >> 24]; /* (int)(l 16 f0) */
        }
    }
    cur->gamma = AmbeDgQ12[b[2]] + prev->gamma / 2;

    int32_t Gm[9] = {0};
    for (int i = 0; i < 3; i++) {
        Gm[2 + i] = AmbePRBA24Q12[b[3]][i];
    }
    for (int i = 0; i < 4; i++) {
        Gm[5 + i] = AmbePRBA58Q12[b[4]][i];
    }
    int32_t Ri[9];
    for (int i = 1; i <= 8; i++) {
        int32_t s = 0;
        for (int m = 1; m <= 8; m++) { /* cos((m-1)(i-1/2) pi / 8) = (m-1)(2i-1)/32 turns */
            s += ((m == 1 ? 1 : 2) * Gm[m] * cos_q12((uint32_t)((m - 1) * (2 * i - 1)) << 27)) >> 12;
        }
        Ri[i] = s;
    }
    int32_t Cik[5][18];
    memset(Cik, 0, sizeof(Cik));
    for (int i = 1; i <= 4; i++) {
        Cik[i][1] = (Ri[2 * i - 1] + Ri[2 * i]) / 2;
        Cik[i][2] = ((Ri[2 * i - 1] - Ri[2 * i]) * 1448) >> 12; /* 1 / (2 sqrt 2) */
    }
    int Ji[5];
    for (int i = 1; i <= 4; i++) {
        Ji[i] = AmbeLmprbl[L][i - 1];
    }
    for (int i = 1; i <= 4; i++) {
        const short *h = hoc_row(i, b[4 + i]);
        for (int k = 3; k <= Ji[i] && k <= 6; k++) {
            Cik[i][k] = h[k - 3];
        }
    }
    /* inverse DCT of each block: the prediction residuals Tl, Q12 */
    int32_t Tl[57], tsum = 0;
    int n = 1;
    for (int i = 1; i <= 4; i++) {
        int ji = Ji[i];
        uint32_t step = ji ? (1u << 30) / ji : 0; /* (k-1)(j-1/2)/(2 ji) turns = (k-1)(2j-1) step */
        for (int j = 1; j <= ji; j++) {
            int32_t s = 0;
            for (int k = 1; k <= ji; k++) {
                s += ((k == 1 ? 1 : 2) * Cik[i][k] * cos_q12((uint32_t)((k - 1) * (2 * j - 1)) * step)) >> 12;
            }
            Tl[n++] = s;
            tsum += s;
        }
    }
    /* log2 Ml predicted from the previous frame */
    if (cur->L > prev->L) {
        for (int l = prev->L + 1; l <= cur->L; l++) {
            prev->Ml[l] = prev->Ml[prev->L];
            prev->log2Ml[l] = prev->log2Ml[prev->L];
        }
    }
    prev->log2Ml[0] = prev->log2Ml[1];
    prev->Ml[0] = prev->Ml[1];
    int Lc = cur->L;
    int32_t pred[57], psum = 0;
    for (int l = 1; l <= Lc; l++) {
        int t = prev->L * l;
        int k = t / Lc;
        int32_t frac = ((t % Lc) << 12) / Lc; /* Q12 */
        int32_t a = prev->log2Ml[k], c = prev->log2Ml[k + 1];
        pred[l] = a + (((c - a) * frac) >> 12);
        psum += pred[l];
    }
    int32_t sum43 = ((psum / Lc) * 2662) >> 12; /* 0.65 x the mean */
    int32_t big_gamma = cur->gamma - log2_q12((uint32_t)Lc) / 2 - tsum / Lc;
    for (int l = 1; l <= Lc; l++) {
        int32_t v = Tl[l] + ((pred[l] * 2662) >> 12) - sum43 + big_gamma;
        cur->log2Ml[l] = v;
        /* Ml = exp(0.693 log2Ml) = 2^(0.99979 log2Ml), as mbelib */
        uint32_t m = exp2_q12(v - ((v * 881) >> 22), 6);
        cur->Ml[l] = (int32_t)(m > 0xFFFFFF ? 0xFFFFFF : m); /* 2^18: far above 16-bit PCM */
    }
    return 0;
}
