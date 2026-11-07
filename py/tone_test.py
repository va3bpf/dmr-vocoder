"""Tone tests of the decoder: the standard 1031 Hz test tone, a sweep of all single tones and the DTMF digits.

  python tone_test.py

1. The published 1031.25 Hz tone frame (NXDN TS 1-A 7.2.1, -24.9 dBFS) for 5 s: frequency, level, SINAD, THD.
2. Every single tone index 5..122 (156..3812 Hz) at the same amplitude for 1 s: frequency error, level against
   the coded level (shows the output high-pass), SINAD.
3. DTMF 0..9: both components and their levels.
Measured after 0.3 s (settled), on the decoder output before clipping. Writes vocoder/out/tone_test.png.
The plain mbelib port is run on the 1031 Hz frames for comparison (it takes tone frames as erasures).
"""

import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ambe_decoder_mbelib
import ambe_tone
from ambe_decoder import Decoder
from ambe_frame import encode_frame
from compare import FS, OUT

TEST_1031 = bytes.fromhex("CEA8FE83ACC458200A")
SKIP = int(0.3 * FS)


def bits72(b):
    return [(x >> (7 - i)) & 1 for x in b for i in range(8)]


def run(frames72, dec=None):
    dec = dec or Decoder()
    return np.concatenate([dec.decode_frame72(f)[0] for f in frames72])


def sine_fit(x, f):
    """Least-squares sine at f Hz (with DC): -> (peak amplitude, residual)."""
    n = np.arange(len(x))
    A = np.stack([np.cos(2 * np.pi * f / FS * n), np.sin(2 * np.pi * f / FS * n), np.ones(len(x))], axis=1)
    c, *_ = np.linalg.lstsq(A, x, rcond=None)
    return math.hypot(c[0], c[1]), x - A @ c


def peak_freq(x, near=None, span=60.0):
    """Frequency of the largest spectral peak (near a given frequency), 64k-point zero-padded Hann FFT."""
    nfft = 1 << 16
    s = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft))
    f = np.fft.rfftfreq(nfft, 1 / FS)
    if near is not None:
        s = np.where(np.abs(f - near) < span, s, 0)
    k = int(np.argmax(s))
    a, b, c = np.log(s[k - 1:k + 2] + 1e-12)
    return f[k] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])


def measure(x, f_nom):
    f = peak_freq(x, f_nom)
    a, res = sine_fit(x, f)
    sinad = 10 * math.log10(np.sum(x ** 2) / np.sum(res ** 2))
    harm = [sine_fit(x, k * f)[0] for k in range(2, 6) if k * f < FS / 2]
    thd = 20 * math.log10(math.sqrt(sum(h * h for h in harm)) / a) if harm else float("nan")
    return f, a, sinad, thd


def dbfs(a):
    return 20 * math.log10(a / 32767)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT, exist_ok=True)
    report = []

    # 1. 1031 Hz test tone
    frames = [bits72(TEST_1031)] * 250
    y = run(frames)
    ym = run(frames, ambe_decoder_mbelib.Decoder())
    f, a, sinad, thd = measure(y[SKIP:], 1031.25)
    want = dbfs(ambe_tone.peak(92))
    report += ["1031 Hz 测试音（NXDN 公开向量，5 秒）",
               "  频率 %.2f Hz（标称 1031.25，误差 %+.3f Hz）" % (f, f - 1031.25),
               "  电平 %.2f dBFS 峰值（编码 %.2f，差 %+.2f dB）" % (dbfs(a), want, dbfs(a) - want),
               "  SINAD %.1f dB，THD %.1f dB" % (sinad, thd),
               "  mbelib 原样：RMS %.1f（把单音帧当擦除帧，输出静音）" % np.sqrt(np.mean(ym ** 2))]

    # 2. single tone sweep
    sweep = []
    for i in range(5, 123):
        fr = [encode_frame(ambe_tone.build(i, 92))] * 50
        x = run(fr)[SKIP:]
        fi, ai, si, _ = measure(x, 31.25 * i)
        sweep.append((31.25 * i, fi - 31.25 * i, dbfs(ai) - want, si))
    sweep = np.array(sweep)
    pass_band = sweep[:, 0] >= 300
    report += ["单音扫频 156～3812 Hz（编号 5～122，电平 %.1f dBFS）" % want,
               "  频率误差最大 %.3f Hz" % np.abs(sweep[:, 1]).max(),
               "  电平偏差 300 Hz 以上 %+.2f～%+.2f dB；156 Hz %+.1f dB，250 Hz %+.1f dB（120 Hz 高通）"
               % (sweep[pass_band, 2].min(), sweep[pass_band, 2].max(), sweep[0, 2], sweep[3, 2]),
               "  SINAD 最低 %.1f dB（%.0f Hz），中位 %.1f dB"
               % (sweep[:, 3].min(), sweep[np.argmin(sweep[:, 3]), 0], np.median(sweep[:, 3]))]

    # 3. DTMF
    report.append("DTMF（编号 128～137 = 0～9，各 1 秒）")
    for digit, i in zip("0123456789", (128, 129, 130, 131, 132, 133, 134, 135, 136, 137)):
        x = run([encode_frame(ambe_tone.build(i, 92))] * 50)[SKIP:]
        hi, lo = ambe_tone.DUAL[i]
        fh, fl = peak_freq(x, hi, 40), peak_freq(x, lo, 40)
        n = np.arange(len(x))
        A = np.stack([np.cos(2 * np.pi * g / FS * n) for g in (fh, fl)] +
                     [np.sin(2 * np.pi * g / FS * n) for g in (fh, fl)], axis=1)
        c, *_ = np.linalg.lstsq(A, x, rcond=None)
        ah, al = math.hypot(c[0], c[2]), math.hypot(c[1], c[3])
        res = x - A @ c
        report.append("  %s：%.1f / %.1f Hz（标称 %d / %d），电平 %+.2f / %+.2f dB，SINAD %.1f dB"
                      % (digit, fh, fl, hi, lo, dbfs(ah) - want, dbfs(al) - want,
                         10 * math.log10(np.sum(x ** 2) / np.sum(res ** 2))))

    print("\n".join(report))

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    t = np.arange(len(y)) / FS
    ax.plot(t, y, lw=0.6, color="C2", label="我们")
    ax.plot(t, ym, lw=0.6, color="C7", label="mbelib 原样")
    ax.set_xlim(0, 0.1)
    ax.set_title("1031 Hz 测试音：前 100 ms 波形（含起始）", loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    seg = y[SKIP:]
    s = 20 * np.log10(np.abs(np.fft.rfft(seg * np.blackman(len(seg)))) / (len(seg) * 0.42 / 2) / 32767 + 1e-12)
    ax.plot(np.fft.rfftfreq(len(seg), 1 / FS), s, lw=0.6, color="C2")
    ax.set_ylim(-140, 0)
    ax.set_title("1031 Hz 测试音：频谱（dBFS，Blackman 窗）", loc="left", fontsize=10)
    ax.set_xlabel("Hz")
    ax.grid(alpha=0.3)
    for row, (col, title, ylim) in enumerate(((2, "单音扫频：电平偏差（dB，相对编码电平）", None),
                                              (3, "单音扫频：SINAD（dB）", None),
                                              (1, "单音扫频：频率误差（Hz）", None))):
        ax = fig.add_subplot(gs[1 + row, :])
        ax.plot(sweep[:, 0], sweep[:, col], ".-", ms=3, lw=0.8, color="C%d" % row)
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 4000)
    ax.set_xlabel("Hz")
    fig.text(0.5, 0.995, "  |  ".join(report[:4]), ha="center", va="top", fontsize=9)
    png = os.path.join(OUT, "tone_test.png")
    fig.savefig(png, dpi=80, bbox_inches="tight")
    print("wrote", os.path.relpath(png))


if __name__ == "__main__":
    main()
