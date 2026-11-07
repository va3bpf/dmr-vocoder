"""AMBE+2 3600x2450 decoder, the plain mbelib port kept as the baseline (ambe_decoder.py is the tuned one):
49 vocoder bits -> 160 samples (20 ms at 8 kHz).

A straight port of mbelib (ambe3600x2450.c mbe_decodeAmbe2450Parms / mbe_processAmbe2450Dataf and mbelib.c),
kept close to the C so that the two can be compared line by line. Improving on it comes later.
"""

import math

import numpy as np

import mbe_tables as T
from ambe_frame import decode_frame

N = 160
WS = np.array(T.Ws)


class Parms:
    """mbe_parms. Arrays are indexed by harmonic l = 1..56 (one spare slot, mbelib reads one past L)."""

    def __init__(self):
        self.w0 = 0.09378
        self.L = 30
        self.gamma = 0.0
        self.repeat = 0
        self.Vl = np.zeros(58, dtype=int)
        self.Ml = np.zeros(58)
        self.log2Ml = np.zeros(58)
        self.PHIl = np.zeros(58)
        self.PSIl = np.full(58, math.pi / 2)

    def copy_from(self, o):
        self.w0, self.L, self.gamma, self.repeat = o.w0, o.L, o.gamma, o.repeat
        self.Vl = o.Vl.copy()
        self.Ml = o.Ml.copy()
        self.log2Ml = o.log2Ml.copy()
        self.PHIl = o.PHIl.copy()
        self.PSIl = o.PSIl.copy()


def _bits(d, idx):
    n = 0
    for i in idx:
        n = (n << 1) | d[i]
    return n


# Where the quantizer indices b0..b8 sit in the 49 bits, MSB first
B_BITS = (
    (0, 1, 2, 3, 37, 38, 39),              # b0  7 bits  fundamental frequency (120..127: erasure, silence, tone)
    (4, 5, 6, 7, 35),                      # b1  5 bits  voiced / unvoiced
    (8, 9, 10, 11, 36),                    # b2  5 bits  gain
    (12, 13, 14, 15, 16, 17, 18, 19, 40),  # b3  9 bits  PRBA 2..4
    (20, 21, 22, 23, 41, 42, 43),          # b4  7 bits  PRBA 5..8
    (24, 25, 26, 27, 44),                  # b5  5 bits  higher order coefficients, block 1
    (28, 29, 30, 45),                      # b6  4 bits  block 2
    (31, 32, 33, 46),                      # b7  4 bits  block 3
    (34, 47, 48),                          # b8  3 bits  block 4
)


def unpack_b(d49):
    return [_bits(d49, idx) for idx in B_BITS]


def decode_parms(d49, cur, prev):
    """mbe_decodeAmbe2450Parms. Returns 0 (voice / silence), 2 (erasure) or 3 (tone). Changes prev like mbelib."""
    b = unpack_b(d49)
    cur.repeat = prev.repeat
    b0 = b[0]
    silence = False
    if 120 <= b0 <= 123:
        return 2
    if b0 in (124, 125):
        silence = True
        cur.w0 = 2 * math.pi / 32
        f0 = 1 / 32
        L = 14
        cur.L = 14
        cur.Vl[1:L + 1] = 0
    elif b0 >= 126:
        return 3
    else:
        f0 = T.AmbeW0table[b0]
        cur.w0 = f0 * 2 * math.pi
        L = int(T.AmbeLtable[b0])
        cur.L = L
    unvc = 0.2046 / math.sqrt(cur.w0)

    if not silence:
        for l in range(1, L + 1):
            cur.Vl[l] = T.AmbeVuv[b[1]][int(l * 16.0 * f0)]

    cur.gamma = T.AmbeDg[b[2]] + 0.5 * prev.gamma

    Gm = [0.0] * 9
    Gm[2:5] = T.AmbePRBA24[b[3]]
    Gm[5:9] = T.AmbePRBA58[b[4]]
    Ri = [0.0] * 9
    for i in range(1, 9):
        Ri[i] = sum((1 if m == 1 else 2) * Gm[m] * math.cos(math.pi * (m - 1) * (i - 0.5) / 8) for m in range(1, 9))

    rconst = 1 / (2 * math.sqrt(2))
    Cik = [[0.0] * 18 for _ in range(5)]
    for i in range(1, 5):
        Cik[i][1] = 0.5 * (Ri[2 * i - 1] + Ri[2 * i])
        Cik[i][2] = rconst * (Ri[2 * i - 1] - Ri[2 * i])

    Ji = [0] + list(T.AmbeLmprbl[L])
    hoc = (None, T.AmbeHOCb5[b[5]], T.AmbeHOCb6[b[6]], T.AmbeHOCb7[b[7]], T.AmbeHOCb8[b[8]])
    for i in range(1, 5):
        for k in range(3, Ji[i] + 1):
            Cik[i][k] = 0.0 if k > 6 else hoc[i][k - 3]

    # Inverse DCT of each block gives the prediction residuals Tl
    Tl = [0.0]
    for i in range(1, 5):
        ji = Ji[i]
        for j in range(1, ji + 1):
            Tl.append(sum((1 if k == 1 else 2) * Cik[i][k] * math.cos(math.pi * (k - 1) * (j - 0.5) / ji)
                          for k in range(1, ji + 1)))

    # log2 Ml predicted from the previous frame
    if cur.L > prev.L:
        prev.Ml[prev.L + 1:cur.L + 1] = prev.Ml[prev.L]
        prev.log2Ml[prev.L + 1:cur.L + 1] = prev.log2Ml[prev.L]
    prev.log2Ml[0] = prev.log2Ml[1]
    prev.Ml[0] = prev.Ml[1]

    Lc = cur.L
    l = np.arange(1, Lc + 1)
    flokl = (prev.L / Lc) * l
    intkl = flokl.astype(int)
    deltal = flokl - intkl
    pred = (1 - deltal) * prev.log2Ml[intkl] + deltal * prev.log2Ml[intkl + 1]
    sum43 = 0.65 / Lc * pred.sum()
    big_gamma = cur.gamma - 0.5 * math.log2(Lc) - sum(Tl[1:Lc + 1]) / Lc
    cur.log2Ml[1:Lc + 1] = np.array(Tl[1:Lc + 1]) + 0.65 * pred - sum43 + big_gamma
    m = np.exp(0.693 * cur.log2Ml[1:Lc + 1])
    cur.Ml[1:Lc + 1] = np.where(cur.Vl[1:Lc + 1] == 1, m, unvc * m)
    return 0


def spectral_amp_enhance(cur):
    L = cur.L
    l = np.arange(1, L + 1)
    M = cur.Ml[1:L + 1]
    Rm0 = np.sum(M * M)
    Rm1 = np.sum(M * M * np.cos(cur.w0 * l))
    R2m0, R2m1 = Rm0 * Rm0, Rm1 * Rm1
    with np.errstate(divide="ignore", invalid="ignore"):
        Wl = np.sqrt(M) * np.power(0.96 * math.pi * ((R2m0 + R2m1) - 2 * Rm0 * Rm1 * np.cos(cur.w0 * l))
                                   / (cur.w0 * Rm0 * (R2m0 - R2m1)), 0.25)
    new = M.copy()
    for i in range(L):
        if M[i] == 0 or 8 * (i + 1) <= L:
            continue
        if Wl[i] > 1.2:
            new[i] = 1.2 * M[i]
        elif Wl[i] < 0.5:
            new[i] = 0.5 * M[i]
        else:
            new[i] = Wl[i] * M[i]
    s = np.sum(new * new)
    cur.Ml[1:L + 1] = new * (1.0 if s == 0 else math.sqrt(Rm0 / s))


def synthesize(cur, prev, rng, uvquality=3):
    """mbe_synthesizeSpeechf: overlap-add of the previous and the current frame, Ws is the 2N+1 window."""
    uvthreshold = 2700 * math.pi / 4000
    uvsine = 1.3591409 * math.e
    uvrand = 2.0
    qfactor = 1 / math.e if uvquality == 1 else math.log(uvquality) / uvquality
    uvstep = 1.0 / uvquality
    uvoffset = uvstep * (uvquality - 1) / 2

    num_uv = int(np.sum(cur.Vl[1:cur.L + 1] == 0))
    cw0, pw0 = cur.w0, prev.w0
    out = np.zeros(N)
    n = np.arange(N)
    w_cur, w_prev = WS[0:N], WS[N:2 * N]

    if cur.L > prev.L:
        maxl = cur.L
        prev.Ml[prev.L + 1:maxl + 1] = 0
        prev.Vl[prev.L + 1:maxl + 1] = 1
    else:
        maxl = prev.L
        cur.Ml[cur.L + 1:maxl + 1] = 0
        cur.Vl[cur.L + 1:maxl + 1] = 1

    for l in range(1, 57):
        cur.PSIl[l] = prev.PSIl[l] + (pw0 + cw0) * (l * N / 2)
        if l <= cur.L // 4:
            cur.PHIl[l] = cur.PSIl[l]
        else:
            cur.PHIl[l] = cur.PSIl[l] + num_uv * _rand_phase(rng) / cur.L

    def multisine(w0, l):
        wl = w0 * l
        rphase = [_rand_phase(rng) for _ in range(uvquality)]
        s = np.zeros(N)
        for i in range(uvquality):
            s += np.cos(w0 * n * (l + i * uvstep - uvoffset) + rphase[i])
            if wl > uvthreshold:
                s += (wl - uvthreshold) * uvrand * rng.random(N)
        return s

    for l in range(1, maxl + 1):
        cv, pv = cur.Vl[l], prev.Vl[l]
        if cv == 0 and pv == 1:
            out += w_prev * prev.Ml[l] * np.cos(pw0 * l * n + prev.PHIl[l])
            out += multisine(cw0, l) * uvsine * w_cur * cur.Ml[l] * qfactor
        elif cv == 1 and pv == 0:
            out += w_cur * cur.Ml[l] * np.cos(cw0 * l * (n - N) + cur.PHIl[l])
            out += multisine(pw0, l) * uvsine * w_prev * prev.Ml[l] * qfactor
        elif cv == 1 or pv == 1:
            out += w_prev * prev.Ml[l] * np.cos(pw0 * l * n + prev.PHIl[l])
            out += w_cur * cur.Ml[l] * np.cos(cw0 * l * (n - N) + cur.PHIl[l])
        else:
            out += multisine(pw0, l) * uvsine * w_prev * prev.Ml[l] * qfactor
            out += multisine(cw0, l) * uvsine * w_cur * cur.Ml[l] * qfactor
    return out


def _rand_phase(rng):
    return rng.random() * 2 * math.pi - math.pi


class Decoder:
    def __init__(self, seed=0, uvquality=3):
        self.rng = np.random.default_rng(seed)
        self.uvquality = uvquality
        self._init()

    def _init(self):
        self.cur, self.prev, self.prev_enh = Parms(), Parms(), Parms()

    def decode_bits(self, d49, errs2=0):
        """mbe_processAmbe2450Dataf: 49 bits -> 160 float samples (mbelib scale) and the frame kind."""
        cur, prev = self.cur, self.prev
        bad = decode_parms(d49, cur, prev)
        if bad in (2, 3):
            cur.repeat = 0
        elif errs2 > 3:
            cur.copy_from(prev)
            cur.repeat += 1
        else:
            cur.repeat = 0
        if bad == 0 and cur.repeat <= 3:
            prev.copy_from(cur)
            spectral_amp_enhance(cur)
            out = synthesize(cur, self.prev_enh, self.rng, self.uvquality)
            self.prev_enh.copy_from(cur)
            return out, "R" if cur.repeat else "V"
        self._init()
        return np.zeros(N), {2: "E", 3: "T"}.get(bad, "M")

    def decode_frame72(self, bits72):
        d, e0, e1 = decode_frame(bits72)
        out, kind = self.decode_bits(d, e0 + e1)
        return out, kind, d, e0 + e1


def to_pcm16(x):
    """mbe_floattoshort: gain 7, clip to +-32760."""
    return np.clip(x * 7, -32760, 32760).astype(np.int16)
