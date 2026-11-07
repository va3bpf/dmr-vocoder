"""Decode a DVSI rate 33 (DMR) test vector with our decoder and compare it with DVSI's own decode.

  python compare.py [name] [--start S] [--len S]

name is a vector in vocoder/data/dvsi/r33 (dam, clean, noisy, dam_e5_hd, ...; default dam). Writes to vocoder/out/:
  <name>.png       input, DVSI decode, the plain mbelib port and our tuned decoder: waveforms, spectrograms,
                   average spectra, frame parameters
  <name>_*.wav     the three signals, to listen to
--start / --len pick a time window for the waveform and spectrogram panels (seconds).
"""

import argparse
import os
import wave

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ambe_decoder import Decoder, to_pcm16, unpack_b
import ambe_decoder_mbelib
import mbe_tables as T

HERE = os.path.dirname(os.path.abspath(__file__))
DVSI = os.path.join(HERE, "..", "data", "dvsi")
OUT = os.path.join(HERE, "..", "out")
FS = 8000

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_pcm(path):
    return np.fromfile(path, "<i2").astype(float)


def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(np.clip(x, -32768, 32767).astype("<i2").tobytes())


def read_bits(path):
    d = open(path, "rb").read()
    return [[(x >> (7 - i)) & 1 for x in d[k:k + 9] for i in range(8)] for k in range(0, len(d), 9)]


def delay(ref, x, max_lag=800):
    """Lag of x behind ref in samples, by cross-correlation of the envelopes (our phases are random)."""
    a = np.abs(ref) - np.abs(ref).mean()
    b = np.abs(x) - np.abs(x).mean()
    n = min(len(a), len(b))
    c = [np.dot(a[:n - lag], b[lag:n]) for lag in range(max_lag)]
    return int(np.argmax(c))


def avg_spectrum(x, nfft=256):
    frames = [x[i:i + nfft] * np.hanning(nfft) for i in range(0, len(x) - nfft, nfft // 2)]
    p = np.mean([np.abs(np.fft.rfft(f)) ** 2 for f in frames], axis=0)
    return np.fft.rfftfreq(nfft, 1 / FS), 10 * np.log10(p + 1e-9)


def frame_db(x):
    """Log energy per 20 ms frame."""
    n = len(x) // 160
    return 10 * np.log10(np.mean(x[:n * 160].reshape(n, 160) ** 2, axis=1) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="dam")
    ap.add_argument("--start", type=float, default=0)
    ap.add_argument("--len", type=float, default=0)
    a = ap.parse_args()
    base = a.name.split("_")[0]

    src = read_pcm(os.path.join(DVSI, base + ".pcm"))
    dvsi = read_pcm(os.path.join(DVSI, "r33", a.name + ".pcm"))
    frames = read_bits(os.path.join(DVSI, "r33", a.name + ".bit"))

    dec = Decoder()
    ours, kinds, b, errs = [], [], [], []
    for f in frames:
        out, kind, d, e = dec.decode_frame72(f)
        ours.append(to_pcm16(out).astype(float))
        kinds.append(kind)
        b.append(unpack_b(d))
        errs.append(e)
    ours = np.concatenate(ours)
    b = np.array(b)
    base = ambe_decoder_mbelib.Decoder()
    mbelib = np.concatenate([ambe_decoder_mbelib.to_pcm16(base.decode_frame72(f)[0]).astype(float) for f in frames])

    os.makedirs(OUT, exist_ok=True)
    for tag, x in (("input", src), ("dvsi", dvsi), ("mbelib", mbelib), ("ours", ours)):
        write_wav(os.path.join(OUT, "%s_%s.wav" % (a.name, tag)), x)

    lag_dvsi = delay(src, dvsi)
    from evaluate import spectral_lag
    lag_ours = spectral_lag(dvsi, ours)
    kinds_count = {k: kinds.count(k) for k in sorted(set(kinds))}
    rms = lambda x: np.sqrt(np.mean(x ** 2))
    # Our output shifted by lag_ours lines up with DVSI's (lag_ours < 0: ours is ahead)
    ours_al = np.concatenate([np.zeros(max(0, -lag_ours)), ours[max(0, lag_ours):]])
    n = min(len(dvsi), len(ours_al))
    e_dvsi, e_ours = frame_db(dvsi[:n]), frame_db(ours_al[:n])
    speech = e_dvsi > e_dvsi.max() - 40
    print("%s: %d frames %s, Golay corrections %d" % (a.name, len(frames), kinds_count, sum(errs)))
    print("delay: DVSI behind input %d samples, ours behind DVSI %d samples" % (lag_dvsi, lag_ours))
    print("RMS: input %.0f  DVSI %.0f  ours %.0f (ours / DVSI %.2f)" % (rms(src), rms(dvsi), rms(ours), rms(ours) / rms(dvsi)))
    print("frame energy vs DVSI over speech frames: mean %+.1f dB, abs %.1f dB"
          % (np.mean(e_ours[speech] - e_dvsi[speech]), np.mean(np.abs(e_ours[speech] - e_dvsi[speech]))))

    t0 = a.start
    t1 = a.start + a.len if a.len else len(src) / FS
    fig = plt.figure(figsize=(16, 22))
    gs = fig.add_gridspec(11, 2, width_ratios=[4, 1.3], hspace=0.6, wspace=0.12)
    sigs = (("原始输入", src, "C0"), ("DVSI 解码", dvsi, "C1"), ("mbelib 原样移植", mbelib, "C7"),
            ("我们的解码（调整后）", ours, "C2"))
    ax0 = None
    for i, (title, x, col) in enumerate(sigs):
        t = np.arange(len(x)) / FS
        ax = fig.add_subplot(gs[i, 0], sharex=ax0)
        ax0 = ax0 or ax
        ax.plot(t, x, color=col, lw=0.4)
        ax.set_title("波形：" + title, loc="left", fontsize=10)
        ax.set_ylim(-32768, 32767)
        ax = fig.add_subplot(gs[4 + i, 0], sharex=ax0)
        with np.errstate(divide="ignore"):  # digital silence
            ax.specgram(x, NFFT=256, Fs=FS, noverlap=192, cmap="magma", vmin=-40, vmax=60)
        ax.set_title("语谱图（256 点 FFT）：" + title, loc="left", fontsize=10)
        ax.set_ylabel("Hz")

    ft = np.arange(len(b)) * 0.02
    f0 = np.array([T.AmbeW0table[x] * FS if x < 120 else np.nan for x in b[:, 0]])
    ax = fig.add_subplot(gs[8, 0], sharex=ax0)
    ax.plot(ft, f0, ".", ms=2, color="C3")
    ax.set_title("基频 f0（b0 查表，Hz）", loc="left", fontsize=10)
    ax.set_ylim(0, 450)
    ax = fig.add_subplot(gs[9, 0], sharex=ax0)
    ax.plot(ft, b[:, 2], lw=0.6, color="C4", label="增益 b2")
    ax.plot(ft, b[:, 1], lw=0.6, color="C5", label="清浊音 b1（0 = 全浊）")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("增益和清浊音索引", loc="left", fontsize=10)
    ax = fig.add_subplot(gs[10, 0], sharex=ax0)
    ax.plot(ft[:len(e_dvsi)], e_dvsi, lw=0.6, color="C1", label="DVSI")
    ax.plot(ft[:len(e_ours)], e_ours, lw=0.6, color="C2", label="我们")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("每帧能量（dB，已对齐延时）", loc="left", fontsize=10)
    ax.set_xlabel("秒")
    ax0.set_xlim(t0, t1)

    ax = fig.add_subplot(gs[0:4, 1])
    for title, x, col in sigs:
        f, p = avg_spectrum(x)
        ax.plot(f, p, color=col, lw=1, label=title)
    ax.set_title("平均频谱（dB）", fontsize=10)
    ax.set_xlabel("Hz")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[4:8, 1])
    ax.axis("off")
    ax.text(0, 1, "\n".join((
        "测试向量 r33/%s" % a.name,
        "%d 帧，%.1f 秒" % (len(frames), len(frames) * 0.02),
        "帧类型 %s" % kinds_count,
        "（V 语音 R 重复 E 擦除 T 音调 M 静音）",
        "Golay 纠错位数 %d" % sum(errs),
        "",
        "DVSI 比输入延后 %d 样点" % lag_dvsi,
        "我们比 DVSI 延后 %d 样点" % lag_ours,
        "",
        "RMS 输入 %.0f" % rms(src),
        "RMS DVSI %.0f" % rms(dvsi),
        "RMS mbelib %.0f（%.2f 倍 DVSI）" % (rms(mbelib), rms(mbelib) / rms(dvsi)),
        "RMS 我们 %.0f（%.2f 倍 DVSI）" % (rms(ours), rms(ours) / rms(dvsi)),
        "",
        "语音帧能量差（我们 − DVSI）",
        "平均 %+.1f dB，绝对值平均 %.1f dB" % (np.mean(e_ours[speech] - e_dvsi[speech]),
                                         np.mean(np.abs(e_ours[speech] - e_dvsi[speech]))),
    )), va="top", fontsize=10, family="Microsoft YaHei")

    png = os.path.join(OUT, a.name + ".png")
    fig.savefig(png, dpi=80, bbox_inches="tight")
    print("wrote", png)


if __name__ == "__main__":
    main()
