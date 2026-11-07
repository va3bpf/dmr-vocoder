"""Speech analysis for the encoder: 8 kHz PCM -> per 20 ms frame w0, voicing and log2 amplitudes of the harmonics.

  pitch     tracked over frames as in TIA-102.BABA chapter 5 (written from its description): the error E(P) of
            a periodic fit for P = 20..128 in half samples, on the input low-passed to 1 kHz under a 301-sample
            window; look-back (within +-20 % of the last two frames' pitch) against look-ahead (the best path
            through the next two frames, with a check of sub-multiples of the period). Then w0 is refined on a
            fine grid by fitting the harmonic model to the windowed spectrum, and multiples of w0 that fit
            nearly as well are taken (no half pitch).
            PITCH_TRACK = False: per-frame autocorrelation candidates instead, scored by the fit error with a
            penalty for leaving the previous frame's pitch.
  harmonic  l: least-squares fit A_l of the window's spectrum shifted to l w0 over the band (l +- 0.5) w0;
  voicing   per 500 Hz band (the V/UV codebook's resolution): voiced when the fit of its harmonics leaves less
            than VOICED_ERR of their energy
  amplitude voiced: the fitted sinusoid's amplitude; unvoiced: the sinusoid with the band's power. Both are
            divided by OUT_GAIN, the decoder's output gain, so that our decoder plays them back at the input level.

Frame k is analysed around sample 160 k + CENTER of the input (found from DVSI's gain against the input energy).
"""

import math

import numpy as np

FS = 8000
N = 160
NW = 256            # analysis window
NFFT = 1024         # zero-padded spectrum
PITCH_CENTER = -200 # where the pitch is estimated (-280 matches DVSI's b0 more often but sounds no better)
REFIT_SPAN = 0      # refine w0 within +-REFIT_SPAN at CENTER (0: off; only matters when PITCH_CENTER != CENTER)
CENTER = -200       # frame k is centred at input sample 160 k - 200, where DVSI centres its frame k
PMIN, PMAX = 20, 128
VOICED_ERR = 0.8    # a 500 Hz band is voiced when the harmonic fit leaves less than this part of its energy
VOICING_HYSTERESIS = 2.0  # threshold x this for a band voiced in the last frame, / this for an unvoiced one (1: off)
REFINE_FMAX = 3000   # Hz: the harmonics below this refine w0
QUIET_FULL = -25.0  # dB below the speech peak: from here down the voicing threshold falls ...
QUIET_NONE = -42.0  # ... to nothing here (fitted to the stock encoder's voicing by level, then widened by PESQ on md380)
SUBMULT_MAXERR = 1.0  # the multiples are tried only when the chosen pitch fits better than this (relative error):
                      # in a poorly periodic frame (a voicing edge) they all fit about as badly, and 3 x w0 won there
HOLD_E = 0.8          # a frame whose E(P) is above this keeps the last frame's pitch (9: off)
VOICE_MAX_E = 0.85    # a frame whose E(P) is above this has no voiced band (9: off): no octave errors at voicing edges
VOICE_E_FULL = 0.4   # the voicing threshold falls from full at this E(P) of the frame's pitch ...
VOICE_E_NONE = 0.8   # ... to nothing here (consonants and syllable edges, E 0.5..0.9, were voiced: a muddy sound
                     # through the stock decoder); 9: off
LONG_P = 95          # a frame whose pitch period is at least this many samples (a low f0) ...
LONG_MAX_E = 0.3     # ... has no voiced band above this E(P): E is measured over few periods there and the harmonics
                     # are too close to tell from noise, a fricative came out as a 65..85 Hz buzz
SUBMULT_MAX_E = 9.0   # ... and only when the tracker's E(P) is below this (a poorly periodic frame keeps its period)
HALF_ODD_DB = 6      # w0 / 2 is taken when its odd harmonics (400..2000 Hz, between the harmonics of w0) are within
                     # this many dB of its even ones (99: off): the true fundamental of a band-passed voice is weak,
                     # the tracker took its second harmonic for the pitch
HALF_NEAR_PREV = True   # ... and only when w0 / 2 is nearer the last frame's pitch than w0 (it undoes an octave jump)
SUBMULT_JUMP = False  # the tracker's sub-multiple check counts the jump from the last frame (PITCH_JUMP_COST)
SUBMULT_TOL = 0.05   # a multiple of w0 is taken if its fit error is at most this much worse
OUT_GAIN = 1.7      # the decoder gain the amplitudes are scaled for: 1.7 gives the stock decoder the stock encoder's loudness

PITCH_NW = 256      # window of the pitch refinement (samples)
PAD = 1024          # zeros before and after the input
AT_QUANTIZED_W0 = True

import mbe_tables as _T
W0_TABLE = np.array(_T.AmbeW0table)
L_TABLE = [int(x) for x in _T.AmbeLtable]
PITCH_TRACK = True
NI = 301            # pitch window
P_GRID = np.arange(PMIN, PMAX + 0.25, 0.5)
LOG_P = np.log(P_GRID)
TRACK_PMAX = 100     # periods above this count as aperiodic (E 1) in the tracker: E(P) divides by 1 - P sum(w^4),
                     # about 0.2 at P 128, and came out near 0 for noise (a 64 Hz pitch, every band voiced)
LOWBAND_MIN = 0.03  # a frame with less of its energy below 1 kHz (the pitch band) than this has no periodicity: the
                    # E(P) of the residue of a fricative found a 62 Hz pitch with every band voiced (0: off)
PITCH_JUMP_COST = 1.0  # added to E per unit of |log| change of the period between frames (0: TIA-102.BABA's tracker)
WI = np.hamming(NI)
WI2 = WI ** 2 / np.sum(WI ** 2)  # w_I^2, sum 1
SUM_W4 = np.sum(WI2 ** 2)
_t = np.arange(-10, 11)
LPF = np.sinc(2 * 1000 / FS * _t) * np.hamming(21)  # 21-tap low-pass at 1 kHz for the pitch
LPF /= LPF.sum()

WIN = np.hamming(NW)
_WRES = 64          # window transform tabulated at NFFT * _WRES points
_WT = np.fft.fft(WIN, NFFT * _WRES)


def window_at(dw):
    """Window transform at frequency offsets dw (rad / sample), linear interpolation in the fine table."""
    pos = (np.asarray(dw) % (2 * np.pi)) / (2 * np.pi) * len(_WT)
    i = np.floor(pos).astype(int)
    f = pos - i
    return _WT[i % len(_WT)] * (1 - f) + _WT[(i + 1) % len(_WT)] * f


def _load_c():
    """bin/vocoder_c.dll (c/analysis.c, built by c/build.sh), or None: then the Python below does the work."""
    import ctypes
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "vocoder_c.dll")
    if not os.path.exists(path) or os.environ.get("VOCODER_NO_C"):
        return None
    lib = ctypes.CDLL(os.path.abspath(path))
    d = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    i, f = ctypes.c_int, ctypes.c_double
    lib.harmonic_fit.argtypes = [d, d, i, i, d, d, i, f, i, d, d, d, d]
    lib.harmonic_fit.restype = None
    return lib


_C = _load_c()
_WT_RE = np.ascontiguousarray(_WT.real)
_WT_IM = np.ascontiguousarray(_WT.imag)


def _c_fit(S, w0, L, wt=None):
    sr = np.ascontiguousarray(S.real)
    si = np.ascontiguousarray(S.imag)
    wr, wi = (_WT_RE, _WT_IM) if wt is None else wt
    ar, ai, err, en = (np.zeros(L + 1) for _ in range(4))
    _C.harmonic_fit(sr, si, len(S), NFFT, wr, wi, len(wr), float(w0), int(L), ar, ai, err, en)
    return L, ar + 1j * ai, err, en


_WINDOWS = {}


def pitch_window(n):
    """Hamming window of n samples and its transform table (re, im) for the pitch refinement's harmonic fit."""
    if n not in _WINDOWS:
        w = np.hamming(n)
        t = np.fft.fft(w, NFFT * _WRES)
        _WINDOWS[n] = (w, (np.ascontiguousarray(t.real), np.ascontiguousarray(t.imag)))
    return _WINDOWS[n]


class Analyzer:
    def __init__(self, x):
        self.x = np.concatenate([np.zeros(PAD), np.asarray(x, dtype=float), np.zeros(PAD)])
        self.xl = np.convolve(self.x, LPF, "same")
        self.prev_p = None
        self.track_e = 0.0  # E(P) of the tracker's last choice
        self.last_w0 = None
        self.e_of = {}  # frame -> E(P) of its pitch
        self._pwt = None  # the window table of the pitch refinement while pitch() runs
        self.prev_bands = np.zeros(8, dtype=bool)  # last frame's voicing per 500 Hz band (hysteresis)
        self._E = {}
        # tracker state: the last two frames' period (grid index) and their E
        self.hist = [(int(np.argmin(np.abs(P_GRID - 100))), 0.0)] * 2

    def E(self, k):
        """E(P) for P_GRID at frame k (0: perfectly periodic, about 1: no periodicity)."""
        if k not in self._E:
            c = PAD + N * k + PITCH_CENTER
            s = self.xl[c - NI // 2:c - NI // 2 + NI]
            sw = s * WI2
            num0 = np.sum(s * s * WI2)
            raw = self.x[c - NI // 2:c - NI // 2 + NI]
            if num0 < 1e-6 or num0 < LOWBAND_MIN * np.sum(raw * raw * WI2):
                self._E[k] = np.ones(len(P_GRID))
            else:
                r = np.correlate(sw, sw, "full")[NI - 1:]
                t = np.arange(len(r))
                e = np.empty(len(P_GRID))
                for i, P in enumerate(P_GRID):
                    lags = P * np.arange(1, int((NI // 2) / P) + 1)
                    tot = r[0] + 2 * np.sum(np.interp(lags, t, r))
                    e[i] = (num0 - P * tot) / (num0 * (1 - P * SUM_W4))
                e[P_GRID > TRACK_PMAX] = 1.0
                self._E[k] = e
        return self._E[k]

    def track(self, k):
        """Pitch period (samples) of frame k: look-back against look-ahead."""
        E0, E1, E2 = self.E(k), self.E(k + 1), self.E(k + 2)
        n = len(P_GRID)
        lo = np.searchsorted(P_GRID, 0.8 * P_GRID - 1e-9)
        hi = np.searchsorted(P_GRID, 1.2 * P_GRID + 1e-9)
        a = PITCH_JUMP_COST

        def best_from(cost, j):
            """min over the +-20 % around P_GRID[j] of cost + a |log P - log P_j|, and where."""
            r = cost[lo[j]:hi[j]] + a * np.abs(LOG_P[lo[j]:hi[j]] - LOG_P[j])
            m = int(np.argmin(r))
            return r[m], lo[j] + m

        # look-back
        (i1, e1), (i2, e2) = self.hist
        cB, iB = best_from(E0, i1)
        ceB = cB + e1 + e2
        # look-ahead: the best continuation through frames k+1 and k+2 for every P0
        m2 = np.array([best_from(E2, j)[0] for j in range(n)])
        c12 = E1 + m2
        ceF = E0 + np.array([best_from(c12, i)[0] for i in range(n)])
        iF = int(np.argmin(ceF))
        # sub-multiples of the look-ahead period that are nearly as good (with SUBMULT_JUMP, counting the jump from the
        # last frame's period too: a good E alone took half or a third of a steady period)
        cF = ceF[iF] + (a * abs(LOG_P[iF] - LOG_P[i1]) if SUBMULT_JUMP else 0)
        for d in range(int(P_GRID[iF] / PMIN), 1, -1):
            j = int(np.argmin(np.abs(P_GRID - P_GRID[iF] / d)))
            c = ceF[j] + (a * abs(LOG_P[j] - LOG_P[i1]) if SUBMULT_JUMP else 0)
            r = c / max(cF, 1e-9)
            if (c <= 0.85 and r <= 1.7) or (c <= 0.4 and r <= 3.5) or (c <= 0.05 and r <= 5.0):
                iF = j
                break
        i = iB if ceB <= 0.48 or ceB <= ceF[iF] + a * abs(LOG_P[iF] - LOG_P[i1]) else iF
        self.hist = [(i, E0[i]), (i1, e1)]
        self.track_e = E0[i]
        return P_GRID[i]

    def segment(self, k, n=NW, center=CENTER):
        c = PAD + N * k + center
        return self.x[c - n // 2:c - n // 2 + n]

    def period(self, k):
        """Coarse pitch period candidates (samples) with their normalized autocorrelation."""
        s = self.segment(k, 2 * NW, PITCH_CENTER)
        s = s - s.mean()
        r = np.correlate(s, s, "full")[len(s) - 1:]
        e = np.cumsum(s[::-1] ** 2)[::-1]  # energy of the overlapping parts
        nccf = r / np.sqrt(max(e[0], 1e-9) * np.maximum(e, 1e-9))
        nccf[:PMIN] = -1
        nccf[PMAX + 1:] = -1
        cands = []
        for p in range(PMIN + 1, PMAX):
            if nccf[p] > nccf[p - 1] and nccf[p] >= nccf[p + 1] and nccf[p] > 0.2:
                a, b, c = nccf[p - 1], nccf[p], nccf[p + 1]
                cands.append((p + 0.5 * (a - c) / (a - 2 * b + c), b))
        return cands

    def spectrum(self, k, center=CENTER):
        s = self.segment(k, NW, center)
        return np.fft.rfft((s - s.mean()) * WIN, NFFT)

    def fit(self, S, w0, lmax=None, wt=None):
        """Harmonic fit of spectrum S at w0: -> (L, A[1..L] complex, err[1..L], energy[1..L]). wt: the transform
        table of another window than WIN (pitch_window(); needs the C fit)."""
        L = int(0.4627 / (w0 / (2 * np.pi))) if lmax is None else lmax
        if _C is not None:
            return _c_fit(S, w0, L, wt)
        assert wt is None, "another window needs bin/vocoder_c.dll"
        wk = np.arange(len(S)) * 2 * np.pi / NFFT
        A = np.zeros(L + 1, dtype=complex)
        err = np.zeros(L + 1)
        en = np.zeros(L + 1)
        for l in range(1, L + 1):
            band = (wk >= (l - 0.5) * w0) & (wk < (l + 0.5) * w0)
            if not band.any():
                continue
            Sb = S[band]
            Wb = window_at(wk[band] - l * w0)
            a = np.vdot(Wb, Sb) / np.vdot(Wb, Wb).real
            A[l] = a
            en[l] = np.sum(np.abs(Sb) ** 2)
            err[l] = np.sum(np.abs(Sb - a * Wb) ** 2)
        return L, A, err, en

    def fit_error(self, S, q):
        """Relative error of the harmonic fit below 3 kHz for period q (samples)."""
        _, _, err, en = self.fit(S, 2 * np.pi / q, max(1, int(REFINE_FMAX / (FS / q))), self._pwt)
        return err.sum() / max(en.sum(), 1e-9)

    def refine(self, S, p, span=0.03):
        """-> (error, w0): the period near p with the least fit error. A 0.5 % grid over +-span, then halving
        steps down to 0.03 % and a parabola through the last three points."""
        n = max(3, int(round(span / 0.005)) * 2 + 1)
        e, q = min((self.fit_error(S, q), q) for q in np.linspace(p * (1 - span), p * (1 + span), n))
        step = 0.0025 * p
        for _ in range(4):
            e, q = min((e, q), (self.fit_error(S, q - step), q - step), (self.fit_error(S, q + step), q + step))
            step /= 2
        el, er = self.fit_error(S, q - step), self.fit_error(S, q + step)
        den = el - 2 * e + er
        if den > 0:
            q += step * min(1.0, max(-1.0, 0.5 * (el - er) / den))
        return min(e, el, er), 2 * np.pi / q

    def frame_energy_db(self, k):
        """Power of frame k's window at CENTER, dB of 16-bit PCM (for the voice activity detector)."""
        s = self.segment(k)
        return 10 * math.log10(np.mean((s - s.mean()) ** 2 * WIN ** 2) / np.mean(WIN ** 2) + 1e-3)

    def noise_amplitudes(self, k):
        """log2 amplitudes of the 14 bands of a silence frame (w0 = 2 pi / 32), as unvoiced bands."""
        _, _, _, en = self.fit(self.spectrum(k), 2 * np.pi / 32, 14)
        amp = np.sqrt(4 * en[1:] / (NFFT * np.sum(WIN ** 2))) / OUT_GAIN
        return np.log2(np.maximum(amp, 1e-3))

    def frame(self, k, level=0.0, w0=None):
        """-> (w0, voiced[1..L] bool, log2 amplitudes[1..L], energy[1..L] (weights for the V/UV quantizer),
        log2 amplitudes as voiced[1..L], as unvoiced[1..L]).

        level: the frame's energy against the speech peak (dB, <= 0). Quiet frames are rarely voiced (as the stock
        encoder has it): the voicing threshold falls to 0 from QUIET_FULL down to QUIET_NONE dB.
        w0: the fundamental to use (pitch() otherwise).

        With AT_QUANTIZED_W0 the harmonics are fitted at the quantized w0 and its L (the ones the decoder makes)."""
        if w0 is None:
            w0 = self.pitch(k)
        return self.harmonics(k, w0, level)

    def pitch(self, k):
        """The fundamental of frame k (rad / sample): tracked, refined, no half pitch."""
        win, self._pwt = pitch_window(PITCH_NW)
        if PITCH_NW == NW:
            self._pwt = None
        s = self.segment(k, PITCH_NW, PITCH_CENTER)
        S = np.fft.rfft((s - s.mean()) * win, NFFT)
        cands = [(self.track(k), 1.0)] if PITCH_TRACK else self.period(k)
        if not cands:
            p = self.prev_p or 50.0
            cands = [(p, 0.0)]
        scored = []
        for p, r in cands:
            e, w0 = self.refine(S, p)
            # a fit is only as good as its error; prefer the previous pitch, and short periods (no sub-octave)
            cost = e - 0.15 * r
            if self.prev_p is not None:
                cost += 0.2 * abs(math.log((2 * np.pi / w0) / self.prev_p))
            scored.append((cost, w0, e))
        cost, w0, e = min(scored)
        # Half the true pitch fits almost as well (every other "harmonic" is a real one, the bands between are
        # nearly empty), so take a multiple of w0 whenever it fits nearly as well
        periodic = e < SUBMULT_MAXERR and (not PITCH_TRACK or self.track_e < SUBMULT_MAX_E)
        for m in (3, 2) if periodic else ():
            p = 2 * np.pi / w0 / m
            if p >= PMIN:
                ek, wk = self.refine(S, p)
                if ek < e + SUBMULT_TOL:
                    e, w0 = ek, wk
                    break
        if HALF_ODD_DB < 99 and 4 * np.pi / w0 <= TRACK_PMAX:
            h = w0 / 2
            n = int(2000 / (FS * h / (2 * np.pi)))
            _, a, _, _ = self.fit(S, h, n, self._pwt)
            f = np.arange(n + 1) * FS * h / (2 * np.pi)
            p2 = np.abs(a) ** 2 * (f >= 400)
            odd, even = np.sum(p2[1::2]), np.sum(p2[2::2])
            near = (not HALF_NEAR_PREV or self.last_w0 is not None
                    and abs(math.log(h / self.last_w0)) < abs(math.log(w0 / self.last_w0)))
            if near and even > 0 and odd > even * 10 ** (-HALF_ODD_DB / 10):
                w0 = h
        if PITCH_TRACK and self.track_e > HOLD_E and self.last_w0 is not None:
            w0 = self.last_w0  # no periodicity to measure: keep the last pitch rather than a chance one
        self.prev_p = 2 * np.pi / w0
        self.last_w0 = w0
        self.e_of[k] = self.track_e if PITCH_TRACK else 0.0  # the voicing of frame k looks at it (harmonics())
        return w0

    def harmonics(self, k, w0, level=0.0):
        """Voicing and amplitudes of frame k's harmonics of w0 (see frame()). With VOICING_HYSTERESIS a band voiced in
        the frame before needs less to stay voiced, an unvoiced one more to become voiced."""
        # the harmonics are fitted at CENTER: let w0 follow the pitch there, a little
        S = self.spectrum(k)
        if REFIT_SPAN:
            w0 = self.refine(S, 2 * np.pi / w0, REFIT_SPAN)[1]
        if AT_QUANTIZED_W0:
            b0 = int(np.argmin(np.abs(np.log(W0_TABLE) - math.log(w0 / (2 * np.pi)))))
            w0 = 2 * np.pi * W0_TABLE[b0]
            L, A, err, en = self.fit(S, w0, L_TABLE[b0])
        else:
            L, A, err, en = self.fit(S, w0)
        # Voicing per 500 Hz band, as the V/UV codebook has it: the fit error of the band's harmonics together
        band = (np.arange(1, L + 1) * w0 / (2 * np.pi) * 16).astype(int)
        e_b = np.bincount(band, weights=err[1:], minlength=8)
        n_b = np.bincount(band, weights=en[1:], minlength=8)
        quiet = min(1.0, max(0.0, (level - QUIET_NONE) / (QUIET_FULL - QUIET_NONE)))
        e = self.e_of.get(k, 0.0)
        if e > (LONG_MAX_E if 2 * np.pi / w0 >= LONG_P else VOICE_MAX_E):
            quiet = 0.0  # a frame without periodicity has no voiced band
        if VOICE_E_NONE < 9:
            quiet *= min(1.0, max(0.0, (VOICE_E_NONE - e) / (VOICE_E_NONE - VOICE_E_FULL)))
        thr = np.full(8, VOICED_ERR * quiet)
        if VOICING_HYSTERESIS:
            thr *= np.where(self.prev_bands, VOICING_HYSTERESIS, 1 / VOICING_HYSTERESIS)
        vb = e_b[:8] < thr * n_b[:8]
        self.prev_bands = vb
        voiced = vb[np.minimum(band, 7)] & (en[1:] > 0)
        # voiced: the fitted sinusoid (S = a/2 e^(j phi) W(w - l w0) for a cos of amplitude a); unvoiced: Parseval,
        # the band's share of the power is 2 en / (NFFT sum(w^2)), a sinusoid of that power has amplitude
        # sqrt(2 * power)
        amp_v = np.maximum(2 * np.abs(A[1:]) / OUT_GAIN, 1e-3)
        amp_u = np.maximum(np.sqrt(4 * en[1:] / (NFFT * np.sum(WIN ** 2))) / OUT_GAIN, 1e-3)
        lv, lu = np.log2(amp_v), np.log2(amp_u)
        return w0, voiced, np.where(voiced, lv, lu), en[1:], lv, lu
