"""How close our decode is to DVSI's, in numbers: spectra per frame, aligned, grouped by frame kind.

  python evaluate.py [name ...]      (default: dam clean noisy)

For each 20 ms frame both outputs get a 256-point Hann spectrum on the same samples (ours shifted by the delay
found against DVSI). Reported, over frames where DVSI is within 50 dB of its loudest frame:
  LSD   log spectral distance, RMS over 0..4 kHz of the dB difference, mean over frames (lower is better)
  LSD0  the same after taking out the mean level difference (shape only, not the gain)
  dE    frame energy difference ours - DVSI, mean and mean absolute (dB)
  bands mean dB difference ours - DVSI in 500 Hz bands, for all-voiced, mixed and all-unvoiced frames
"""

import os
import sys

import numpy as np

from ambe_decoder import Decoder, to_pcm16
from compare import DVSI, read_bits, read_pcm, delay

NFFT = 256
EDGES = np.arange(0, 4001, 500)


def decode_file(name, **opts):
    dec = Decoder(**opts)
    out, vfrac = [], []
    for f in read_bits(os.path.join(DVSI, "r33", name + ".bit")):
        x, kind, d, e = dec.decode_frame72(f)
        out.append(to_pcm16(x).astype(float))
        L = dec.cur.L
        vfrac.append(dec.cur.Vl[1:L + 1].mean() if kind in "VR" else np.nan)
    return np.concatenate(out), np.array(vfrac)


def spectra(x, starts):
    w = np.hanning(NFFT)
    return np.array([20 * np.log10(np.abs(np.fft.rfft(x[s:s + NFFT] * w)) + 1) for s in starts])


def spectral_lag(dvsi, ours, lo=-100, hi=200):
    """The shift of ours against DVSI with the lowest mean LSD (envelope correlation is off by tens of samples).
    Negative: our output is ahead of DVSI's."""
    starts = np.arange(max(0, -lo), min(len(dvsi), len(ours) - hi) - NFFT, 480)
    sd = spectra(dvsi, starts)
    act = sd.max(axis=1) > sd.max() - 60

    def cost(lag):
        return np.sqrt(np.mean((spectra(ours, starts + lag) - sd)[act] ** 2, axis=1)).mean()

    best = min(range(lo, hi, 8), key=cost)
    return min(range(best - 7, best + 8), key=cost)


def evaluate(name, verbose=True, **opts):
    dvsi = read_pcm(os.path.join(DVSI, "r33", name + ".pcm"))
    ours, vfrac = decode_file(name, **opts)
    lag = spectral_lag(dvsi, ours)
    # Frame k of the bit stream sits at DVSI samples ~k*160 - lag ... (ours[lag + n] lines up with dvsi[n])
    n = min(len(dvsi), len(ours) - lag) - NFFT
    starts = np.arange(max(0, -lag), n, 160)
    sd, so = spectra(dvsi, starts), spectra(ours, starts + lag)
    ed = 10 * np.log10(np.array([np.mean(dvsi[s:s + NFFT] ** 2) for s in starts]) + 1)
    eo = 10 * np.log10(np.array([np.mean(ours[s + lag:s + lag + NFFT] ** 2) for s in starts]) + 1)
    act = ed > ed.max() - 50
    # Which frame's parameters dominate the window starting at s: our frame index (s + lag + NFFT/2) // 160
    fidx = np.minimum((starts + lag + NFFT // 2) // 160, len(vfrac) - 1)
    vf = vfrac[fidx]
    diff = so - sd
    lsd = np.sqrt(np.mean(diff ** 2, axis=1))
    freqs = np.fft.rfftfreq(NFFT, 1 / 8000)
    lsd0 = np.sqrt(np.mean((diff - diff[act].mean()) ** 2, axis=1))
    res = {"lag": lag, "lsd": lsd[act].mean(), "lsd0": lsd0[act].mean(), "dE": (eo - ed)[act].mean(),
           "|dE|": np.abs(eo - ed)[act].mean()}
    groups = (("voiced", act & (vf == 1)), ("mixed", act & (vf > 0) & (vf < 1)), ("unvoiced", act & (vf == 0)),
              ("quiet", ~act))
    rows = {}
    for g, m in groups:
        if m.sum() == 0:
            continue
        rows[g] = (m.sum(), [diff[m][:, (freqs >= a) & (freqs < b)].mean() for a, b in zip(EDGES[:-1], EDGES[1:])],
                   lsd[m].mean())
    res["groups"] = rows
    if verbose:
        print("%-8s lag %3d  LSD %5.2f dB  LSD0 %5.2f dB  dE %+5.1f dB  |dE| %4.1f dB"
              % (name, lag, res["lsd"], res["lsd0"], res["dE"], res["|dE|"]))
        print("          %-9s %5s %6s  " % ("frames", "n", "LSD") + " ".join("%5d" % e for e in EDGES[:-1]) + "  Hz")
        for g, (cnt, bands, l) in rows.items():
            print("          %-9s %5d %6.2f  " % (g, cnt, l) + " ".join("%+5.1f" % v for v in bands))
    return res


if __name__ == "__main__":
    for name in sys.argv[1:] or ("dam", "clean", "noisy"):
        evaluate(name)
