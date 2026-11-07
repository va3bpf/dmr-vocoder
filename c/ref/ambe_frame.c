/*
 * DMR AMBE+2 frame layer: 72 channel bits <-> 49 vocoder bits (py/ambe_frame.py).
 * C0: Golay(24,12) (fr[0][0] overall parity, even), C1: Golay(23,12) XOR a PN sequence seeded by C0's data,
 * C2 11 bits, C3 14 bits; interleaved as dsd's rW/rX/rY/rZ.
 */
#include "ambe.h"

static const uint8_t W[36] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2,
                              0, 2, 0, 2, 0, 2, 0, 2};
static const uint8_t X[36] = {23, 10, 22, 9, 21, 8, 20, 7, 19, 6, 18, 5, 17, 4, 16, 3, 15, 2, 14, 1, 13, 0, 12, 10,
                              11, 9, 10, 8, 9, 7, 8, 6, 7, 5, 6, 4};
static const uint8_t Y[36] = {0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 0, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3,
                              1, 3, 1, 3, 1, 3, 1, 3};
static const uint8_t Z[36] = {5, 3, 4, 2, 3, 1, 2, 0, 1, 13, 0, 12, 22, 11, 21, 10, 20, 9, 19, 8, 18, 7, 17, 6,
                              16, 5, 15, 4, 14, 3, 13, 2, 12, 1, 11, 0};

/* Golay(23,12) parity of each data bit, data MSB first (mbelib golayGenerator) */
static const uint16_t GOLAY_GEN[12] = {0x63a, 0x31d, 0x7b4, 0x3da, 0x1ed, 0x6cc, 0x366, 0x1b3, 0x6e3, 0x54b, 0x49f,
                                       0x475};

/* the B_BITS of py/ambe_decoder.py: where b0..b8 sit, MSB first */
static const uint8_t B_LEN[9] = {7, 5, 5, 9, 7, 5, 4, 4, 3};
static const uint8_t B_BITS[9][9] = {
    {0, 1, 2, 3, 37, 38, 39},
    {4, 5, 6, 7, 35},
    {8, 9, 10, 11, 36},
    {12, 13, 14, 15, 16, 17, 18, 19, 43},
    {20, 21, 22, 23, 40, 41, 42},
    {24, 25, 26, 27, 44},
    {28, 29, 30, 45},
    {31, 32, 33, 46},
    {34, 47, 48},
};

static uint16_t golay23_parity(uint16_t data)
{
    uint16_t p = 0;
    for (int i = 0; i < 12; i++) {
        if (data & (0x800 >> i)) {
            p ^= GOLAY_GEN[i];
        }
    }
    return p;
}

static uint32_t golay23_encode(uint16_t data)
{
    return ((uint32_t)data << 11) | golay23_parity(data);
}

static int popcount(uint32_t v)
{
    int n = 0;
    for (; v; v &= v - 1) {
        n++;
    }
    return n;
}

/* The code is perfect: each of the 2048 syndromes belongs to exactly one error pattern of weight <= 3 */
static uint32_t golay_err[2048];
static int golay_ready;

static void golay_add(uint32_t e)
{
    golay_err[golay23_parity((uint16_t)(e >> 11)) ^ (e & 0x7FF)] = e;
}

static void golay_init(void)
{
    golay_add(0);
    for (int a = 0; a < 23; a++) {
        golay_add(1u << a);
        for (int b = a + 1; b < 23; b++) {
            golay_add((1u << a) | (1u << b));
            for (int c = b + 1; c < 23; c++) {
                golay_add((1u << a) | (1u << b) | (1u << c));
            }
        }
    }
    golay_ready = 1;
}

static uint16_t golay23_decode(uint32_t block, int *errs)
{
    if (!golay_ready) {
        golay_init();
    }
    uint32_t e = golay_err[golay23_parity((uint16_t)(block >> 11)) ^ (block & 0x7FF)];
    *errs += popcount(e);
    return (uint16_t)((block ^ e) >> 11);
}

/* The 23 bits XOR-ed onto C1, for fr[1][22] down to fr[1][0] */
static void pn(uint16_t seed12, uint8_t out[23])
{
    uint32_t pr = 16u * seed12;
    for (int i = 0; i < 23; i++) {
        pr = (173 * pr + 13849) & 0xFFFF;
        out[i] = (uint8_t)(pr >> 15);
    }
}

void ambe_encode_frame(const uint8_t d49[49], uint8_t out9[9])
{
    uint8_t fr[4][24] = {{0}};
    uint16_t c0 = 0, c1 = 0;
    for (int i = 0; i < 12; i++) {
        c0 = (uint16_t)((c0 << 1) | d49[i]);
        c1 = (uint16_t)((c1 << 1) | d49[12 + i]);
    }
    uint32_t g0 = golay23_encode(c0), g1 = golay23_encode(c1);
    int par = 0;
    for (int j = 1; j <= 23; j++) {
        fr[0][j] = (g0 >> (j - 1)) & 1;
        par ^= fr[0][j];
    }
    fr[0][0] = (uint8_t)par;
    for (int j = 0; j <= 22; j++) {
        fr[1][j] = (g1 >> j) & 1;
    }
    uint8_t p[23];
    pn(c0, p);
    for (int k = 0; k < 23; k++) {
        fr[1][22 - k] ^= p[k];
    }
    for (int i = 0; i < 11; i++) {
        fr[2][10 - i] = d49[24 + i];
    }
    for (int i = 0; i < 14; i++) {
        fr[3][13 - i] = d49[35 + i];
    }
    for (int i = 0; i < 9; i++) {
        out9[i] = 0;
    }
    for (int i = 0; i < 36; i++) {
        int b0 = fr[W[i]][X[i]], b1 = fr[Y[i]][Z[i]];
        out9[(2 * i) >> 3] |= (uint8_t)(b0 << (7 - ((2 * i) & 7)));
        out9[(2 * i + 1) >> 3] |= (uint8_t)(b1 << (7 - ((2 * i + 1) & 7)));
    }
}

int ambe_decode_frame(const uint8_t in9[9], uint8_t d49[49])
{
    uint8_t fr[4][24] = {{0}};
    for (int i = 0; i < 36; i++) {
        fr[W[i]][X[i]] = (in9[(2 * i) >> 3] >> (7 - ((2 * i) & 7))) & 1;
        fr[Y[i]][Z[i]] = (in9[(2 * i + 1) >> 3] >> (7 - ((2 * i + 1) & 7))) & 1;
    }
    int errs = 0;
    uint32_t v = 0;
    for (int j = 23; j >= 1; j--) {
        v = (v << 1) | fr[0][j];
    }
    uint16_t c0 = golay23_decode(v, &errs);
    uint8_t p[23];
    pn(c0, p);
    for (int k = 0; k < 23; k++) {
        fr[1][22 - k] ^= p[k];
    }
    v = 0;
    for (int j = 22; j >= 0; j--) {
        v = (v << 1) | fr[1][j];
    }
    uint16_t c1 = golay23_decode(v, &errs);
    for (int i = 0; i < 12; i++) {
        d49[i] = (c0 >> (11 - i)) & 1;
        d49[12 + i] = (c1 >> (11 - i)) & 1;
    }
    for (int i = 0; i < 11; i++) {
        d49[24 + i] = fr[2][10 - i];
    }
    for (int i = 0; i < 14; i++) {
        d49[35 + i] = fr[3][13 - i];
    }
    return errs;
}

void ambe_unpack_b(const uint8_t d49[49], int b[9])
{
    for (int f = 0; f < 9; f++) {
        int v = 0;
        for (int i = 0; i < B_LEN[f]; i++) {
            v = (v << 1) | d49[B_BITS[f][i]];
        }
        b[f] = v;
    }
}

void ambe_pack_b(const int b[9], uint8_t d49[49])
{
    for (int f = 0; f < 9; f++) {
        int n = B_LEN[f];
        for (int i = 0; i < n; i++) {
            d49[B_BITS[f][i]] = (uint8_t)((b[f] >> (n - 1 - i)) & 1);
        }
    }
}
