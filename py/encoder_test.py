"""Test the encoder against DVSI's on their rate 33 test vectors (the same input PCM).

  python encoder_test.py [name ...]      (default: dam)

1. Parameters, frame by frame against DVSI's bits (dequantized), after finding the frame offset:
   pitch within 5 % / octave errors, V/UV agreement over harmonics, gain and spectral envelope difference (dB).
2. End to end with the same decoder: our bits -> our decoder against DVSI's bits -> our decoder (LSD, energy).
3. Against the input: our bits and DVSI's bits, each through our decoder, against the original PCM (LSD, STOI,
   PESQ; quality.py).
   Item 2 counts any pitch difference from DVSI as an error, even where ours is the truer one; this does not.
Writes vocoder/out/<name>_enc.png and <name>_enc_ours.wav (our encoder + our decoder).
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ambe_decoder import Decoder, Parms, decode_parms, to_pcm16
from ambe_encoder import encode_pcm
from ambe_frame import decode_frame
from compare import DVSI, FS, OUT, read_bits, read_pcm, write_wav
from evaluate import NFFT, spectra, spectral_lag

DB = 20 * np.log10(2)  # dB per unit of log2 amplitude


def params(frames):
    """Dequantized parameters of each frame: w0, L, voiced[1..L], log2Ml[1..L], gamma."""
    cur, prev = Parms(), Parms()
    out = []
    for f in frames:
        d = decode_frame(f)[0]
        decode_parms(d, cur, prev, unvc_scale=False)
        prev.copy_from(cur)
        L = cur.L
        out.append((cur.w0, L, cur.Vl[1:L + 1].copy(), cur.log2Ml[1:L + 1].copy(), cur.gamma))
    return out


def resample(x, L):
    return np.interp(np.arange(1, L + 1) * len(x) / L, np.arange(1, len(x) + 1), x)


def decode(frames):
    dec = Decoder()
    return np.concatenate([to_pcm16(dec.decode_frame72(f)[0]).astype(float) for f in frames])


def lsd_input(src, y):
    """(LSD, STOI, PESQ) of decoded y against the original: LSD over the input's active frames, at the best lag;
    STOI (intelligibility 0..1) on the aligned signals; PESQ (MOS-LQO, P.862 narrow band; finds the delay itself)."""
    import quality

    lag = spectral_lag(src, y, 0, 800)
    starts = np.arange(0, min(len(src), len(y) - lag) - NFFT, 160)
    diff = spectra(y, starts + lag) - spectra(src, starts)
    e = 10 * np.log10(np.array([np.mean(src[s:s + NFFT] ** 2) for s in starts]) + 1)
    act = e > e.max() - 50
    return (np.sqrt(np.mean(diff ** 2, axis=1))[act].mean(), quality.stoi(src, y, lag), quality.pesq(src, y))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for name in sys.argv[1:] or ("dam",):
        src = read_pcm(os.path.join(DVSI, name + ".pcm"))
        ref = read_bits(os.path.join(DVSI, "r33", name + ".bit"))
        ours, info = encode_pcm(src)
        pr, po = params(ref), params(ours)

        # frame offset: our frame k against DVSI frame k + m, by the pitch agreement on voiced frames
        def agree(m):
            n = 0
            for k in range(len(po)):
                j = k + m
                if 0 <= j < len(pr) and pr[j][2].mean() > 0.5:
                    n += abs(np.log(po[k][0] / pr[j][0])) < 0.05
            return n
        m = max(range(-6, 7), key=agree)

        rows = []
        for k in range(len(po)):
            j = k + m
            if not 0 <= j < len(pr):
                continue
            w0r, Lr, vr, lr, gr = pr[j]
            w0o, Lo, vo, lo, go = po[k]
            rows.append((k, j, w0r, w0o, vr.mean(), np.mean(resample(vo.astype(float), Lr) >= 0.5) if Lo else 0,
                         np.mean((resample(vo.astype(float), Lr) >= 0.5) == vr),
                         (np.mean(lo) - np.mean(lr)) * DB,
                         np.sqrt(np.mean(((resample(lo, Lr) - np.mean(lo)) - (lr - np.mean(lr))) ** 2)) * DB,
                         np.mean(lr)))
        r = np.array(rows)
        loud = r[:, 9] > np.percentile(r[:, 9], 40)  # leave out the pauses
        voiced = loud & (r[:, 4] > 0.5)
        ratio = r[voiced, 3] / r[voiced, 2]
        print("%s: %d frames, our frame k = DVSI frame k%+d" % (name, len(ours), m))
        print("  pitch (DVSI voiced, %d frames): within 5 %% %.1f %%, octave errors (x2 or /2) %.1f %%"
              % (voiced.sum(), 100 * np.mean(np.abs(np.log(ratio)) < 0.05),
                 100 * np.mean(np.abs(np.abs(np.log2(ratio)) - 1) < 0.1)))
        print("  V/UV agreement over harmonics: %.1f %%   voiced fraction DVSI %.2f ours %.2f"
              % (100 * r[loud, 6].mean(), r[loud, 4].mean(), r[loud, 5].mean()))
        print("  gain ours - DVSI: mean %+.1f dB, |.| %.1f dB;  envelope shape RMS %.1f dB"
              % (r[loud, 7].mean(), np.abs(r[loud, 7]).mean(), r[loud, 8].mean()))

        y_ref, y_ours = decode(ref), decode(ours)
        lag = spectral_lag(y_ref, y_ours, -400, 400)
        starts = np.arange(max(0, -lag), min(len(y_ref), len(y_ours) - lag) - NFFT, 160)
        diff = spectra(y_ours, starts + lag) - spectra(y_ref, starts)
        e = 10 * np.log10(np.array([np.mean(y_ref[s:s + NFFT] ** 2) for s in starts]) + 1)
        act = e > e.max() - 50
        print("  end to end (same decoder): LSD %.2f dB, level %+.1f dB (lag %d)"
              % (np.sqrt(np.mean(diff ** 2, axis=1))[act].mean(), diff[act].mean(), lag))
        print("  against the input, our decoder for both: DVSI's bits LSD %.2f dB STOI %.3f PESQ %.3f;"
              "  ours LSD %.2f dB STOI %.3f PESQ %.3f" % (lsd_input(src, y_ref) + lsd_input(src, y_ours)))
        os.makedirs(OUT, exist_ok=True)
        write_wav(os.path.join(OUT, name + "_enc_ours.wav"), y_ours)

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axs = plt.subplots(5, 1, figsize=(16, 15), sharex=True)
        t = r[:, 0] * 0.02
        axs[0].plot(t, r[:, 2] / (2 * np.pi) * FS, ".", ms=3, color="C1", label="DVSI")
        axs[0].plot(t, r[:, 3] / (2 * np.pi) * FS, ".", ms=2, color="C2", label="我们")
        axs[0].set_ylim(0, 420)
        axs[0].set_title("基频 f0（Hz）", loc="left", fontsize=10)
        axs[0].legend(fontsize=8)
        axs[1].plot(t, r[:, 4], lw=0.7, color="C1", label="DVSI")
        axs[1].plot(t, r[:, 5], lw=0.7, color="C2", label="我们")
        axs[1].set_title("浊音谐波比例", loc="left", fontsize=10)
        axs[1].legend(fontsize=8)
        axs[2].plot(t, r[:, 9] * DB, lw=0.7, color="C1", label="DVSI")
        axs[2].plot(t, (r[:, 9] + r[:, 7] / DB) * DB, lw=0.7, color="C2", label="我们")
        axs[2].set_title("平均谱幅度（dB）", loc="left", fontsize=10)
        axs[2].legend(fontsize=8)
        axs[3].plot(t, r[:, 8], lw=0.7, color="C3")
        axs[3].set_title("谱包络形状差（RMS dB，扣除平均电平）", loc="left", fontsize=10)
        for ax, (title, y) in zip(axs[4:], (("我们编码 + 我们解码：波形", y_ours),)):
            ax.plot(np.arange(len(y)) / FS, y / 32768 * 100, lw=0.3, color="C2")
            ax.set_title(title, loc="left", fontsize=10)
        axs[-1].set_xlabel("秒")
        fig.savefig(os.path.join(OUT, name + "_enc.png"), dpi=70, bbox_inches="tight")


if __name__ == "__main__":
    main()
