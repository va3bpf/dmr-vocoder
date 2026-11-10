"""AMBE+2 3600x2450 decoder: 49 vocoder bits -> 160 samples (20 ms at 8 kHz).

Started as a port of mbelib (ambe3600x2450.c mbe_decodeAmbe2450Parms / mbe_processAmbe2450Dataf and mbelib.c,
kept as ambe_decoder_mbelib.py) and tuned towards DVSI's decode of their rate 33 test vectors (evaluate.py):
output gain, unvoiced bands as shaped noise, half-strength spectral enhancement, phase-continuous low harmonics
and a 120 Hz high-pass. Each change is an option of Decoder; the parameter decoding is still mbelib's.
"""

import math

import numpy as np

import mbe_tables as T
import ambe_tone
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


# Where the quantizer indices b0..b8 sit in the 49 bits, MSB first. The LSB of b3 and the 3 LSBs of b4 are where
# DVSI's vocoder puts them (b4 at 40..42, b3 at 43), not where the draft TIA-102.BABA-1 Table 8 and mbelib do
# (b3 at 40, b4 at 41..43): found by go-ambe2 with a radio's DVSI vocoder as a black box, and on DVSI's rate 33
# vectors our decode is closer to DVSI's this way (LSD dam 7.19 -> 6.97, clean 7.50 -> 7.28, noisy 7.65 -> 7.30).
B_BITS = (
    (0, 1, 2, 3, 37, 38, 39),              # b0  7 bits  fundamental frequency (120..127: erasure, silence, tone)
    (4, 5, 6, 7, 35),                      # b1  5 bits  voiced / unvoiced
    (8, 9, 10, 11, 36),                    # b2  5 bits  gain
    (12, 13, 14, 15, 16, 17, 18, 19, 43),  # b3  9 bits  PRBA 2..4
    (20, 21, 22, 23, 40, 41, 42),          # b4  7 bits  PRBA 5..8
    (24, 25, 26, 27, 44),                  # b5  5 bits  higher order coefficients, block 1
    (28, 29, 30, 45),                      # b6  4 bits  block 2
    (31, 32, 33, 46),                      # b7  4 bits  block 3
    (34, 47, 48),                          # b8  3 bits  block 4
)


def unpack_b(d49):
    return [_bits(d49, idx) for idx in B_BITS]


def decode_parms(d49, cur, prev, unvc_scale=True):
    """mbe_decodeAmbe2450Parms. Returns 0 (voice / silence), 2 (erasure) or 3 (tone). Changes prev like mbelib.

    unvc_scale: mbelib scales unvoiced Ml by 0.2046 / sqrt(w0) for its multisine; the noise synthesis does not."""
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
    unvc = 0.2046 / math.sqrt(cur.w0) if unvc_scale else 1.0

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


def spectral_amp_enhance(cur, strength=1.0, voiced_only=False):
    """mbe_spectralAmpEnhance; strength scales the weights in the log domain (1: mbelib, 0: none); voiced_only: the
    unvoiced bands keep their amplitude (and do not take part in the final energy normalisation)."""
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
    Wl = np.power(Wl, strength)
    for i in range(L):
        if M[i] == 0 or 8 * (i + 1) <= L or (voiced_only and not cur.Vl[i + 1]):
            continue
        if Wl[i] > 1.2 ** strength:
            new[i] = 1.2 ** strength * M[i]
        elif Wl[i] < 0.5 ** strength:
            new[i] = 0.5 ** strength * M[i]
        else:
            new[i] = Wl[i] * M[i]
    if voiced_only:
        v = cur.Vl[1:L + 1] == 1
        s = np.sum(new[v] ** 2)
        new[v] *= 1.0 if s == 0 else math.sqrt(np.sum(M[v] ** 2) / s)
        cur.Ml[1:L + 1] = new
        return
    s = np.sum(new * new)
    cur.Ml[1:L + 1] = new * (1.0 if s == 0 else math.sqrt(Rm0 / s))


# A fixed pseudo-random phase offset for each harmonic (phase_disp): harmonics that start in phase add up to a pulse
# train with a high crest factor, which sounds buzzy; go-ambe2 found a DVSI decoder's steady phases fixed but not aligned
PHASE_DISP = np.random.default_rng(12345).uniform(-math.pi, math.pi, 58)
PHASE_DISP[0] = 0.0


def synthesize(cur, prev, rng, uvquality=3, uv_gain=1.0, uv_hf_noise=True, voiced_only=False, phase_interp=False,
               phase_disp=False, interp_max=8):
    """mbe_synthesizeSpeechf: overlap-add of the previous and the current frame, Ws is the 2N+1 window.

    uv_hf_noise: mbelib's extra noise on unvoiced harmonics above 2700 Hz. It is rand() in 0..1, not zero mean,
    so under the window it is a low-frequency thump and broadband noise rather than high-frequency noise.
    voiced_only: leave the unvoiced harmonics out (NoiseSynth makes them).
    phase_interp: a harmonic voiced in both frames with l < 8 and w0 changing by less than 10 % is one sinusoid
    with interpolated amplitude and continuous phase (IMBE eq. 134-138, commented out in mbelib) instead of
    the cross-fade of two sinusoids."""
    uvthreshold = 2700 * math.pi / 4000
    uvsine = 1.3591409 * math.e * uv_gain
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
        if phase_disp:
            cur.PHIl[l] += PHASE_DISP[l]

    def multisine(w0, l):
        wl = w0 * l
        rphase = [_rand_phase(rng) for _ in range(uvquality)]
        s = np.zeros(N)
        for i in range(uvquality):
            s += np.cos(w0 * n * (l + i * uvstep - uvoffset) + rphase[i])
            if uv_hf_noise and wl > uvthreshold:
                s += (wl - uvthreshold) * uvrand * rng.random(N)
        return s

    if voiced_only:
        for l in range(1, maxl + 1):
            if (phase_interp and prev.Vl[l] and cur.Vl[l] and l < interp_max and abs(cw0 - pw0) < 0.1 * cw0):
                # eq. 137, 138: frequency offset so the phase lands on PHIl at n = N
                dphi = cur.PHIl[l] - prev.PHIl[l] - (pw0 + cw0) * l * N / 2
                dw = (dphi - 2 * math.pi * math.floor((dphi + math.pi) / (2 * math.pi))) / N
                theta = prev.PHIl[l] + (pw0 * l + dw) * n + (cw0 - pw0) * l * n * n / (2 * N)
                amp = prev.Ml[l] + n / N * (cur.Ml[l] - prev.Ml[l])
                out += amp * np.cos(theta)
                continue
            if prev.Vl[l]:
                out += w_prev * prev.Ml[l] * np.cos(pw0 * l * n + prev.PHIl[l])
            if cur.Vl[l]:
                out += w_cur * cur.Ml[l] * np.cos(cw0 * l * (n - N) + cur.PHIl[l])
        return out

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


class NoiseSynth:
    """Unvoiced synthesis as in the IMBE / AMBE descriptions: white noise shaped in the frequency domain, kept only
    in the unvoiced bands (l - 0.5) w0 .. (l + 0.5) w0, each band with the power of a sinusoid of amplitude Ml.

    Every frame makes 2N samples of noise from its own parameters; output frame k cross-fades the second half of
    frame k-1's noise into the first half of frame k's with sqrt(Ws), which keeps the power of uncorrelated noise."""

    M = 2 * N

    def __init__(self):
        self.tail = np.zeros(N)

    def frame(self, p, rng, gain):
        M = self.M
        X = rng.standard_normal(M // 2 + 1) + 1j * rng.standard_normal(M // 2 + 1)
        S = np.zeros(M // 2 + 1, dtype=complex)
        k = np.arange(M // 2 + 1) * 2 * np.pi / M  # bin frequency, rad / sample
        for l in range(1, p.L + 1):
            if p.Vl[l] or p.Ml[l] == 0:
                continue
            band = (k >= (l - 0.5) * p.w0) & (k < (l + 0.5) * p.w0) & (k > 0) & (k < np.pi)
            e = np.sum(np.abs(X[band]) ** 2)
            if e == 0:
                continue
            # Parseval for irfft of length M: mean(x^2) = 2 * sum|S_k|^2 / M^2 over 0 < k < M/2
            S[band] = X[band] * math.sqrt((p.Ml[l] ** 2 / 2) * M * M / 2 / e) * gain
        u = np.fft.irfft(S, M)
        out = np.sqrt(WS[N:2 * N]) * self.tail + np.sqrt(WS[0:N]) * u[:N]
        self.tail = u[N:]
        return out


class Biquad:
    """Second-order high-pass (RBJ cookbook), direct form I, state kept across frames."""

    def __init__(self, fc, q=1 / math.sqrt(2), fs=8000):
        w = 2 * math.pi * fc / fs
        alpha = math.sin(w) / (2 * q)
        c = math.cos(w)
        a0 = 1 + alpha
        self.b = ((1 + c) / 2 / a0, -(1 + c) / a0, (1 + c) / 2 / a0)
        self.a = (-2 * c / a0, (1 - alpha) / a0)
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def run(self, x):
        b0, b1, b2 = self.b
        a1, a2 = self.a
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        y = np.empty(len(x))
        for i, v in enumerate(x):
            o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, v, y1, o
            y[i] = o
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return y


def _rand_phase(rng):
    return rng.random() * 2 * math.pi - math.pi


class Decoder:
    """The defaults are the settings closest to DVSI's decoder as md380-emu runs it (the MD-380's, which matches the
    radio's stock decoder): md380 applies mbelib's spectral enhancement at full strength (its per-harmonic gain on
    steady voiced frames = 0.95 x the formula's + 0.66 dB, correlation 0.83), and gain 2.35 / uv_gain 0.77 put the
    voiced, mixed and unvoiced frames within 0.2 dB of its levels. gain=7, enhance=1, uv_hf_noise=True,
    uv_mode="multisine", highpass=0, phase_interp=False is mbelib (ambe_decoder_mbelib.py).

    gain: output scale to 16-bit PCM (mbelib 7); uv_gain: unvoiced level (mbelib 1); enhance: mbelib's spectral
    amplitude enhancement, 0..1 (True = 1); uv_hf_noise: see synthesize(); uv_mode: "multisine" (mbelib) or
    "noise" (NoiseSynth); phase_interp: see synthesize(); highpass: cut-off in Hz of the output high-pass
    (0: none), hp_order: 2 or 4; tones: play tone frames (ambe_tone.py; mbelib takes them as erasures);
    silence_keep: silence frames (b0 124, 125) leave the gain and spectrum prediction to the last voice frame
    (TIA-102.BABA-1 and DVSI's decoder; mbelib updates it: on the stock encoder's Chinese 2.64 -> 2.77 PESQ);
    silence_gain: their level (go-ambe2 measured 0.228 on a DVSI decoder; 0.25 matches the radio's stock decoder,
    which plays them about 11 dB down; 1 as mbelib); phase_disp: a fixed pseudo-random phase offset per harmonic
    (PHASE_DISP; dam 3.185 -> 3.240 PESQ);
    interp_max: harmonics below this get continuous phase and amplitude (phase_interp; 8 in IMBE, 57 for all)."""

    def __init__(self, seed=0, uvquality=3, gain=2.35, uv_gain=0.77, enhance=1.0, uv_hf_noise=False,
                 uv_mode="noise", highpass=120, hp_order=4, phase_interp=True, tones=True, silence_keep=True,
                 silence_gain=0.25, phase_disp=True, interp_max=8, enhance_voiced_only=False, uv_top_db=0.0,
                 v_top_db=0.0):
        self.rng = np.random.default_rng(seed)
        self.uvquality = uvquality
        self.gain = gain
        self.uv_gain = uv_gain
        self.uv_hf_noise = uv_hf_noise
        self.uv_mode = uv_mode
        self.phase_interp = phase_interp
        # 4th order: two 2nd-order sections with the Butterworth Q values
        qs = (1 / math.sqrt(2),) if hp_order == 2 else (0.5412, 1.3066)
        self.hp = [Biquad(highpass, q) for q in qs] if highpass else []
        self.enhance = enhance
        self.tones = tones
        self.silence_keep = silence_keep
        self.silence_gain = silence_gain
        self.phase_disp = phase_disp
        self.interp_max = interp_max
        self.enhance_voiced_only = enhance_voiced_only
        self.uv_top_db = uv_top_db
        self.v_top_db = v_top_db
        self.tone = ambe_tone.ToneSynth()
        self._init()

    def _init(self):
        self.cur, self.prev, self.prev_enh = Parms(), Parms(), Parms()
        self.noise = NoiseSynth()

    def decode_bits(self, d49, errs2=0):
        """mbe_processAmbe2450Dataf: 49 bits -> 160 samples (16-bit PCM scale, not clipped) and the frame kind."""
        if self.tones and ambe_tone.is_tone(d49):
            idx, amp = ambe_tone.parse(d49)
            freqs = ambe_tone.frequencies(idx)
            if freqs is not None and errs2 <= 3:
                self._init()  # the voice parameters start again after the tone, as after an erasure
                return self.tone.frame(freqs, ambe_tone.peak(amp)), "T"
        cur, prev = self.cur, self.prev
        noise = self.uv_mode == "noise"
        silence = unpack_b(d49)[0] in (124, 125)
        keep = None
        if silence and self.silence_keep:
            keep = Parms()
            keep.copy_from(prev)
        bad = decode_parms(d49, cur, prev, unvc_scale=not noise)
        if bad == 2:  # erasure: a lost frame filled in upstream; as DVSI, the last frame again (mbelib mutes it)
            cur.copy_from(prev)
            cur.repeat += 1
            bad = 0
        elif bad == 3:
            cur.repeat = 0
        elif errs2 > 3:
            cur.copy_from(prev)
            cur.repeat += 1
        else:
            cur.repeat = 0
        if bad == 0 and cur.repeat <= 3:
            prev.copy_from(cur if keep is None else keep)
            if silence and self.silence_gain != 1.0:
                cur.Ml *= self.silence_gain
            if self.enhance:
                spectral_amp_enhance(cur, float(self.enhance), self.enhance_voiced_only)
            if self.uv_top_db or self.v_top_db:
                # md380 plays the top 40 % of the harmonics louder, up to uv_top_db (unvoiced) / v_top_db (voiced)
                # at the last one (probed with steady frames, uvprobe / enhprobe)
                L = cur.L
                pos = np.clip((np.arange(1, L + 1) / L - 0.6) / 0.4, 0.0, 1.0)
                top = np.where(cur.Vl[1:L + 1] == 1, self.v_top_db, self.uv_top_db)
                cur.Ml[1:L + 1] *= 10 ** (top * pos / 20)
            out = synthesize(cur, self.prev_enh, self.rng, self.uvquality, self.uv_gain, self.uv_hf_noise, noise,
                             self.phase_interp, self.phase_disp, self.interp_max)
            if noise:
                out = out + self.noise.frame(cur, self.rng, self.uv_gain)
            self.prev_enh.copy_from(cur)
            return out * self.gain + self._tone_off(), "R" if cur.repeat else "V"
        self._init()
        return self._tone_off(), {2: "E", 3: "T"}.get(bad, "M")

    def _tone_off(self):
        """A tone that was playing fades out over this frame."""
        return self.tone.frame([], 0.0) if self.tone.level else np.zeros(N)

    def decode_frame72(self, bits72):
        d, e0, e1 = decode_frame(bits72)
        out, kind = self.decode_bits(d, e0 + e1)
        for f in self.hp:
            out = f.run(out)
        return out, kind, d, e0 + e1


def to_pcm16(x):
    """mbe_floattoshort: clip to +-32760."""
    return np.clip(x, -32760, 32760).astype(np.int16)
