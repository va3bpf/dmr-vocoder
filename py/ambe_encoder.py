"""AMBE+2 3600x2450 encoder: 160 samples (20 ms at 8 kHz) -> 49 vocoder bits.

Written from the decoder (ambe_decoder.decode_parms, mbelib's dequantization) run backwards, not from OP25's
GPL encoder. The quantizer keeps a mirror of the decoder's state (previous log2 Ml, gamma), so its prediction
is the one the receiver makes.

Quantizer, for a frame with fundamental w0, voicing Vl and log2 spectral amplitudes log2Ml (l = 1..L):
  b0  nearest w0 in AmbeW0table (gives L)
  b1  V/UV codebook entry closest to Vl
  b2  gain: gamma = mean(log2Ml) + 0.5 log2 L, quantized as gamma - 0.5 gamma(-1) with AmbeDg
  r   = log2Ml - 0.65 * prediction from the previous frame (the decoder removes the mean of r itself)
      split into 4 blocks (AmbeLmprbl[L]), DCT of each block -> C(i, k)
  b3, b4  PRBA: R(2i-1, 2i) from C(i, 1..2), 8-point DCT G(1..8); G2..4 with PRBA24, G5..8 with PRBA58
  b5..b8  C(i, 3..6) with the HOC codebooks of block i
"""

import math

import numpy as np

import mbe_tables as T
from ambe_decoder import B_BITS, Parms, decode_parms

B0_HYSTERESIS = 0.0   # keep the last pitch step while the new pitch is within this many steps of it (0: off)
PITCH_SMOOTH = False      # smooth_pitch() over the frames: steadier pitch, but on the air the user preferred without
PITCH_SMOOTH_SPAN = 0.04  # neighbours closer than this (relative) are averaged
SILENCE_KEEPS_PREDICTION = True  # silence frames leave the gain and spectrum prediction alone (TIA-102.BABA-1,
                                 # DVSI's decoder as go-ambe2 found; mbelib updates it)
SILENCE_FRAMES = True  # silence frames (b0 = 124) where vad() hears no speech, as the stock encoder sends them
VAD_MARGIN = 15.0   # dB above the noise floor for speech
VAD_MIN_DB = 20.0   # frames below this (dB of 16-bit PCM power, about RMS 10) are never speech
FLOOR_RISE = 0.05   # dB a frame the noise floor may rise
PEAK_DECAY = 0.05   # dB a frame the speech peak may fall (60 dB in 24 s)
VAD_BELOW_PEAK = 45.0  # frames this far below the peak are silence (fitted to the stock encoder: 88 % of frames agree)
VAD_HANGOVER = 1
VAD_LOOKAHEAD = 0
WEIGHTED = True     # PRBA / HOC by the amplitude-weighted error of the reconstruction
MATCHED_AMPLITUDES = False  # amplitudes by the quantized voicing: no better on dam (11.54 against 11.47 dB)
W0 = np.array(T.AmbeW0table) * 2 * math.pi
VUV = np.array(T.AmbeVuv)
DG = np.array(T.AmbeDg)
PRBA24 = np.array(T.AmbePRBA24)
PRBA58 = np.array(T.AmbePRBA58)
HOC = [np.array(T.AmbeHOCb5), np.array(T.AmbeHOCb6), np.array(T.AmbeHOCb7), np.array(T.AmbeHOCb8)]


def pack_b(b):
    """b0..b8 -> 49 bits (inverse of ambe_decoder.unpack_b)."""
    d = [0] * 49
    for v, idx in zip(b, B_BITS):
        n = len(idx)
        for k, i in enumerate(idx):
            d[i] = (v >> (n - 1 - k)) & 1
    return d


def _dct(x):
    """C(k) = 1/J sum_j x(j) cos(pi (k-1) (j-0.5) / J), k = 1..J: the inverse of the decoder's transforms."""
    J = len(x)
    j = np.arange(1, J + 1)
    return np.array([np.sum(x * np.cos(math.pi * k * (j - 0.5) / J)) / J for k in range(J)])


def quantize_w0(w0):
    """Index of the nearest fundamental (in log frequency)."""
    return int(np.argmin(np.abs(np.log(W0) - math.log(w0))))


def quantize_vuv(f0, L, vl, weight=None):
    """V/UV codebook entry whose voicing of harmonics 1..L disagrees least with vl (weighted by weight)."""
    jl = (np.arange(1, L + 1) * 16.0 * f0).astype(int)
    cand = VUV[:, jl]  # 32 x L
    w = np.ones(L) if weight is None else np.asarray(weight)
    err = np.sum(w * (cand != np.asarray(vl)[None, :]), axis=1)
    return int(np.argmin(err))


class Quantizer:
    def __init__(self):
        self.cur, self.prev = Parms(), Parms()
        self.last_b0 = None

    def _weighted(self, b, r, log2ml, L, k3=32, k4=16):
        """PRBA and HOC by the weighted error of the reconstructed residual: sum w_l (T_l - r_l)^2 after taking out
        the means (the decoder takes out the mean of T itself), w_l = (M_l / max M)^0.5. b3 among its k3 best
        unweighted entries, b4 among its k4 best, every pair; then each block's HOC given the rest."""
        A24, A58, H = _recon_matrices(L)
        w = np.power(2.0, 0.5 * (log2ml - log2ml.max()))

        def centre(x):
            return x - x.mean(axis=0)

        rc = r - r.mean()
        t24 = centre(A24 @ PRBA24.T)  # L x 512
        t58 = centre(A58 @ PRBA58.T)  # L x 128
        hoc = sum(centre(H[i] @ HOC[i][b[5 + i]]) for i in range(4))
        # candidates: the best unweighted entries (the same search as before, in the residual domain)
        c3 = np.argsort(np.sum((t24 - (rc - hoc)[:, None]) ** 2, axis=0))[:k3]
        c4 = np.argsort(np.sum((t58 - (rc - hoc)[:, None]) ** 2, axis=0))[:k4]
        best = None
        for i3 in c3:
            e = rc - hoc - t24[:, i3]
            err = np.sum(w[:, None] * (e[:, None] - t58[:, c4]) ** 2, axis=0)
            j = int(np.argmin(err))
            if best is None or err[j] < best[0]:
                best = (err[j], int(i3), int(c4[j]))
        b[3], b[4] = best[1], best[2]
        base = rc - t24[:, b[3]] - t58[:, b[4]]
        for i in range(4):
            if not H[i].any():
                continue
            rest = base - sum(centre(H[m] @ HOC[m][b[5 + m]]) for m in range(4) if m != i)
            cand = centre(H[i] @ HOC[i].T)
            b[5 + i] = int(np.argmin(np.sum(w[:, None] * (rest[:, None] - cand) ** 2, axis=0)))

    def prediction(self, L):
        """The decoder's prediction of log2 Ml from the previous frame, for L harmonics (eq. 40-43)."""
        prev = self.prev
        lm = prev.log2Ml.copy()
        if L > prev.L:
            lm[prev.L + 1:L + 1] = lm[prev.L]
        lm[0] = lm[1]
        l = np.arange(1, L + 1)
        flokl = (prev.L / L) * l
        k = flokl.astype(int)
        d = flokl - k
        return (1 - d) * lm[k] + d * lm[k + 1]

    def quantize(self, w0, vl, log2ml, vuv_weight=None, log2_v=None, log2_u=None, silence=False):
        """w0 (rad / sample), vl and log2ml for harmonics 1..L of w0 -> 49 bits; updates the decoder mirror.

        vl and log2ml may be longer or shorter than the L of the quantized w0: they are resampled to it.
        With log2_v and log2_u (each harmonic's amplitude as voiced and as unvoiced), the amplitudes follow the
        quantized voicing instead of vl ("matched amplitudes").
        silence: a silence frame (b0 = 124, b1 = 16 as the stock encoder sends them): the decoder plays the 14
        bands of w0 = 2 pi / 32 as noise; log2ml are their amplitudes (w0 and vl are ignored)."""
        b = [0] * 9
        if silence:
            b[0], b[1] = 124, 16
            L = 14
            log2ml = _resample(np.asarray(log2ml, dtype=float), L)
        else:
            b[0] = quantize_w0(w0)
            if B0_HYSTERESIS and self.last_b0 is not None:
                # stay on the last frame's pitch step while the new pitch is within B0_HYSTERESIS steps of it:
                # the frame-to-frame jitter of real speech goes, a tone's steady glide does not
                x = -45.368 * math.log2(w0 / (2 * math.pi)) - 195.626  # continuous index (the table follows it)
                if abs(x - self.last_b0) < B0_HYSTERESIS:
                    b[0] = self.last_b0
            f0 = T.AmbeW0table[b[0]]
            L = int(T.AmbeLtable[b[0]])
            vl = _resample(np.asarray(vl, dtype=float), L) >= 0.5
            log2ml = _resample(np.asarray(log2ml, dtype=float), L)
            b[1] = quantize_vuv(f0, L, vl.astype(int), None if vuv_weight is None else _resample(vuv_weight, L))
        if log2_v is not None and MATCHED_AMPLITUDES and not silence:
            vq = VUV[b[1]][(np.arange(1, L + 1) * 16.0 * f0).astype(int)] == 1
            log2ml = np.where(vq, _resample(np.asarray(log2_v, float), L), _resample(np.asarray(log2_u, float), L))

        gamma = np.mean(log2ml) + 0.5 * math.log2(L)
        b[2] = int(np.argmin(np.abs(DG - (gamma - 0.5 * self.prev.gamma))))

        r = log2ml - 0.65 * self.prediction(L)
        Ji = T.AmbeLmprbl[L]
        C, s = [], 0
        for J in Ji:
            C.append(_dct(r[s:s + J]))
            s += J
        R = np.empty(8)
        for i in range(4):
            R[2 * i] = C[i][0] + math.sqrt(2) * C[i][1]
            R[2 * i + 1] = C[i][0] - math.sqrt(2) * C[i][1]
        G = _dct(R)
        b[3] = int(np.argmin(np.sum((PRBA24 - G[1:4]) ** 2, axis=1)))
        b[4] = int(np.argmin(np.sum((PRBA58 - G[4:8]) ** 2, axis=1)))
        for i in range(4):
            n = min(Ji[i], 6) - 2
            if n > 0:
                b[5 + i] = int(np.argmin(np.sum((HOC[i][:, :n] - C[i][2:2 + n]) ** 2, axis=1)))
        if WEIGHTED:
            self._weighted(b, r, log2ml, L)

        self.last_b0 = None if silence else b[0]
        d = pack_b(b)
        keep = Parms()
        keep.copy_from(self.prev)
        decode_parms(d, self.cur, self.prev, unvc_scale=False)
        if silence and SILENCE_KEEPS_PREDICTION:
            self.prev.copy_from(keep)  # as DVSI's decoder: the prediction stays with the last voice frame
        else:
            self.prev.copy_from(self.cur)
        return d, b


DC_POLE = 32512  # (1 - 1/128) in Q15


def dc_block(x):
    """The input's DC taken out (a microphone's bias, 2500 on the DM-32: with the click at the start of a
    transmission it threw the pitch tracker off and every band went unvoiced; DVSI's encoder high-passes its input
    too): y = x - x1 + (1 - 1/128) y1, about 10 Hz, in integers as the C encoders (c/ambe, c/ref dc_block()); x1 starts
    as the first sample."""
    out = np.empty(len(x))
    x = np.clip(np.round(np.asarray(x, dtype=float)), -32768, 32767).astype(np.int64)
    x1, y = (int(x[0]) if len(x) else 0), 0  # the DC there from the start: no step into the blocker
    for i, v in enumerate(x):
        v = int(v)
        y = ((v - x1) << 15) + ((y * DC_POLE) >> 15)
        x1 = v
        out[i] = min(32767, max(-32768, y >> 15))
    return out


def encode_pcm(x):
    """16-bit PCM at 8 kHz -> list of 72-bit frames (and the per-frame analysis, for tests)."""
    from ambe_analysis import Analyzer
    from ambe_frame import encode_frame

    x = dc_block(x)
    an, q = Analyzer(x), Quantizer()
    n = len(x) // 160
    e_db = [an.frame_energy_db(k) for k in range(n)]
    speech = vad(e_db) if SILENCE_FRAMES else [True] * n
    level = peak_level(e_db)
    w0s = smooth_pitch([an.pitch(k) if speech[k] else None for k in range(n)])
    frames, info = [], []
    for k in range(n):
        if speech[k]:
            w0, voiced, log2a, energy, lv, lu = an.frame(k, level[k], w0s[k])
            d, b = q.quantize(w0, voiced, log2a, energy, lv, lu)
        else:
            log2a = an.noise_amplitudes(k)
            w0, voiced, energy = 2 * math.pi / 32, np.zeros(len(log2a), bool), None
            d, b = q.quantize(w0, voiced, log2a, silence=True)
        frames.append(encode_frame(d))
        info.append((w0, voiced, log2a, energy))
    return frames, info


def smooth_pitch(w0s):
    """Median of 3 in log frequency (an outlier between two agreeing neighbours goes), then neighbours within
    PITCH_SMOOTH_SPAN of each other averaged 1:2:1 (the frame-to-frame jitter of the estimate). None: silence."""
    if not PITCH_SMOOTH:
        return w0s
    lg = [None if w is None else math.log(w) for w in w0s]
    med = list(lg)
    for k in range(1, len(lg) - 1):
        if None not in (lg[k - 1], lg[k], lg[k + 1]):
            med[k] = sorted((lg[k - 1], lg[k], lg[k + 1]))[1]
    out = list(med)
    if not PITCH_SMOOTH_SPAN:
        return [None if v is None else math.exp(v) for v in out]
    span = math.log(1 + PITCH_SMOOTH_SPAN)
    for k in range(len(med)):
        if med[k] is None:
            continue
        acc, wt = 2 * med[k], 2.0
        for j in (k - 1, k + 1):
            if 0 <= j < len(med) and med[j] is not None and abs(med[j] - med[k]) < span:
                acc, wt = acc + med[j], wt + 1
        out[k] = acc / wt
    return [None if v is None else math.exp(v) for v in out]


def peak_level(e_db):
    """Each frame's energy against the speech peak (dB, <= 0); the peak as in vad()."""
    peak = None
    out = []
    for e in e_db:
        peak = e if peak is None else max(e, peak - PEAK_DECAY)
        out.append(e - peak)
    return out


def vad(e_db):
    """Speech (True) or silence for each frame, from its energy (dB): a noise floor follows the quietest frames
    (down at once, up by FLOOR_RISE dB a frame) and a peak the loudest (up at once, down by PEAK_DECAY dB a frame);
    speech is VAD_MARGIN dB above the floor and no more than VAD_BELOW_PEAK below the peak. A speech frame keeps
    VAD_HANGOVER frames after it and VAD_LOOKAHEAD before it as speech, so the ends of words stay."""
    floor = peak = None
    raw = []
    for e in e_db:
        floor = e if floor is None else min(e, floor + FLOOR_RISE)
        peak = e if peak is None else max(e, peak - PEAK_DECAY)
        raw.append(e > max(floor + VAD_MARGIN, peak - VAD_BELOW_PEAK, VAD_MIN_DB))
    out = [False] * len(raw)
    for k, r in enumerate(raw):
        if r:
            for j in range(max(0, k - VAD_LOOKAHEAD), min(len(raw), k + VAD_HANGOVER + 1)):
                out[j] = True
    return out


def _recon_matrices(L):
    """The decoder's residual T(1..L) as A24 @ PRBA24[b3] + A58 @ PRBA58[b4] + sum_i H_i @ HOC_i[b(5+i)]
    (ambe_decoder.decode_parms: G -> R -> C(i, 1..2), HOC -> C(i, 3..6), inverse DCT of each block)."""
    if L in _RECON:
        return _RECON[L]
    i8 = np.arange(1, 9)
    m8 = np.arange(1, 9)
    CR = np.array([[(1 if m == 1 else 2) * math.cos(math.pi * (m - 1) * (i - 0.5) / 8) for m in m8] for i in i8])
    rc = 1 / (2 * math.sqrt(2))
    Ji = T.AmbeLmprbl[L]
    # C(i, 1), C(i, 2) from R: rows 2 i, 2 i + 1 of an 8 x 8 matrix
    CfromR = np.zeros((8, 8))
    for i in range(4):
        CfromR[2 * i, 2 * i], CfromR[2 * i, 2 * i + 1] = 0.5, 0.5
        CfromR[2 * i + 1, 2 * i], CfromR[2 * i + 1, 2 * i + 1] = rc, -rc
    Cg = CfromR @ CR  # C(i, 1..2) from G(1..8)
    A = np.zeros((L, 8))
    H = []
    s = 0
    for i, J in enumerate(Ji):
        j = np.arange(1, J + 1)
        idct = np.array([[(1 if k == 1 else 2) * math.cos(math.pi * (k - 1) * (jj - 0.5) / J) for k in range(1, J + 1)]
                         for jj in j]) if J else np.zeros((0, 0))
        if J >= 1:
            A[s:s + J] += np.outer(idct[:, 0], Cg[2 * i])
        if J >= 2:
            A[s:s + J] += np.outer(idct[:, 1], Cg[2 * i + 1])
        n = min(J, 6) - 2
        Hi = np.zeros((L, 4))
        if n > 0:
            Hi[s:s + J, :n] = idct[:, 2:2 + n]
        H.append(Hi)
        s += J
    _RECON[L] = (A[:, 1:4], A[:, 4:8], H)
    return _RECON[L]


_RECON = {}


def _resample(x, L):
    """Values for harmonics 1..len(x) -> harmonics 1..L (linear, ends held)."""
    if len(x) == L:
        return x
    src = np.arange(1, len(x) + 1)
    dst = np.arange(1, L + 1) * len(x) / L
    return np.interp(dst, src, x)


if __name__ == "__main__":
    # Round trip on DVSI's bits: dequantize each frame with a decoder mirror, quantize the parameters again with
    # our quantizer (which has its own mirror) and compare the bits.
    import os
    import sys

    from ambe_frame import decode_frame
    from ambe_decoder import unpack_b
    from compare import DVSI, read_bits

    sys.stdout.reconfigure(encoding="utf-8")
    for name in ("dam", "clean", "noisy"):
        ref_cur, ref_prev = Parms(), Parms()
        q = Quantizer()
        same_bits, same_b = 0, np.zeros(9, int)
        frames = read_bits(os.path.join(DVSI, "r33", name + ".bit"))
        for f in frames:
            d = decode_frame(f)[0]
            decode_parms(d, ref_cur, ref_prev, unvc_scale=False)
            ref_prev.copy_from(ref_cur)
            L = ref_cur.L
            d2, b2 = q.quantize(ref_cur.w0, ref_cur.Vl[1:L + 1], ref_cur.log2Ml[1:L + 1])
            same_bits += d2 == d
            same_b += np.array(unpack_b(d)) == np.array(b2)
        print("%-6s %d 帧：49 bit 全同 %d；各字段相同的帧数 b0..b8 %s"
              % (name, len(frames), same_bits, " ".join(str(x) for x in same_b)))
