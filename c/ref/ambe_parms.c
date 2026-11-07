/*
 * AMBE+2 parameter decoding, 49 bits -> w0, L, voicing and log2 spectral amplitudes (py/ambe_decoder.py
 * decode_parms, after mbelib's mbe_decodeAmbe2450Parms). The encoder runs it too, as its mirror of the decoder.
 */
#include <math.h>
#include <string.h>

#include "ambe.h"
#include "ambe_tables.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void ambe_parms_init(ambe_parms *p)
{
    memset(p, 0, sizeof(*p));
    p->w0 = 0.09378;
    p->L = 30;
    for (int l = 0; l < 58; l++) {
        p->PSIl[l] = M_PI / 2;
    }
}

static const float *hoc_row(int i, int b)
{
    switch (i) {
    case 1:
        return AmbeHOCb5[b];
    case 2:
        return AmbeHOCb6[b];
    case 3:
        return AmbeHOCb7[b];
    default:
        return AmbeHOCb8[b];
    }
}

int ambe_decode_parms(const uint8_t d49[49], ambe_parms *cur, ambe_parms *prev)
{
    int b[9];
    ambe_unpack_b(d49, b);
    cur->repeat = prev->repeat;
    int b0 = b[0], L;
    int silence = 0;
    ambe_real f0;
    if (b0 >= 120 && b0 <= 123) {
        return 2;
    }
    if (b0 == 124 || b0 == 125) {
        silence = 1;
        cur->w0 = 2 * M_PI / 32;
        f0 = 1.0 / 32;
        L = 14;
        cur->L = 14;
        for (int l = 1; l <= L; l++) {
            cur->Vl[l] = 0;
        }
    } else if (b0 >= 126) {
        return 3;
    } else {
        f0 = AmbeW0table[b0];
        cur->w0 = f0 * 2 * M_PI;
        L = AmbeLtable[b0];
        cur->L = L;
    }
    if (!silence) {
        for (int l = 1; l <= L; l++) {
            cur->Vl[l] = AmbeVuv[b[1]][(int)(l * 16.0 * f0)];
        }
    }
    cur->gamma = AmbeDg[b[2]] + 0.5 * prev->gamma;

    ambe_real Gm[9] = {0};
    for (int i = 0; i < 3; i++) {
        Gm[2 + i] = AmbePRBA24[b[3]][i];
    }
    for (int i = 0; i < 4; i++) {
        Gm[5 + i] = AmbePRBA58[b[4]][i];
    }
    ambe_real Ri[9];
    for (int i = 1; i <= 8; i++) {
        ambe_real s = 0;
        for (int m = 1; m <= 8; m++) {
            s += (m == 1 ? 1 : 2) * Gm[m] * cos(M_PI * (m - 1) * (i - 0.5) / 8);
        }
        Ri[i] = s;
    }
    const ambe_real rconst = 1 / (2 * sqrt(2.0));
    ambe_real Cik[5][18];
    memset(Cik, 0, sizeof(Cik));
    for (int i = 1; i <= 4; i++) {
        Cik[i][1] = 0.5 * (Ri[2 * i - 1] + Ri[2 * i]);
        Cik[i][2] = rconst * (Ri[2 * i - 1] - Ri[2 * i]);
    }
    int Ji[5];
    for (int i = 1; i <= 4; i++) {
        Ji[i] = AmbeLmprbl[L][i - 1];
    }
    for (int i = 1; i <= 4; i++) {
        const float *h = hoc_row(i, b[4 + i]);
        for (int k = 3; k <= Ji[i]; k++) {
            Cik[i][k] = k > 6 ? 0.0 : h[k - 3];
        }
    }
    /* inverse DCT of each block: the prediction residuals Tl */
    ambe_real Tl[57];
    int n = 1;
    for (int i = 1; i <= 4; i++) {
        int ji = Ji[i];
        for (int j = 1; j <= ji; j++) {
            ambe_real s = 0;
            for (int k = 1; k <= ji; k++) {
                s += (k == 1 ? 1 : 2) * Cik[i][k] * cos(M_PI * (k - 1) * (j - 0.5) / ji);
            }
            Tl[n++] = s;
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
    ambe_real pred[57], psum = 0, tsum = 0;
    for (int l = 1; l <= Lc; l++) {
        ambe_real flokl = ((ambe_real)prev->L / Lc) * l;
        int intkl = (int)flokl;
        ambe_real deltal = flokl - intkl;
        pred[l] = (1 - deltal) * prev->log2Ml[intkl] + deltal * prev->log2Ml[intkl + 1];
        psum += pred[l];
        tsum += Tl[l];
    }
    ambe_real sum43 = 0.65 / Lc * psum;
    ambe_real big_gamma = cur->gamma - 0.5 * log2((ambe_real)Lc) - tsum / Lc;
    for (int l = 1; l <= Lc; l++) {
        cur->log2Ml[l] = Tl[l] + 0.65 * pred[l] - sum43 + big_gamma;
        cur->Ml[l] = exp(0.693 * cur->log2Ml[l]);
    }
    return 0;
}
