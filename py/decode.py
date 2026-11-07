"""Decode any AMBE .bit file (9 bytes per 20 ms frame) with our decoder and the plain mbelib port.

  python decode.py file.bit [more.bit ...]

For each file writes to vocoder/out/: <stem>_ours.wav, <stem>_mbelib.wav and <stem>.png (waveforms, spectrograms,
pitch, voicing and the Golay corrections per frame, which show the quality of the radio link).
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ambe_decoder_mbelib
import mbe_tables as T
from ambe_decoder import Decoder, to_pcm16, unpack_b
from compare import FS, OUT, read_bits, write_wav


def decode(path):
    frames = read_bits(path)
    dec, base = Decoder(), ambe_decoder_mbelib.Decoder()
    ours, mbelib, b, errs, kinds = [], [], [], [], []
    for f in frames:
        x, kind, d, e = dec.decode_frame72(f)
        ours.append(to_pcm16(x).astype(float))
        mbelib.append(ambe_decoder_mbelib.to_pcm16(base.decode_frame72(f)[0]).astype(float))
        b.append(unpack_b(d))
        errs.append(e)
        kinds.append(kind)
    return np.concatenate(ours), np.concatenate(mbelib), np.array(b), np.array(errs), kinds


def plot(stem, ours, mbelib, b, errs, kinds):
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(7, 1, hspace=0.6)
    sigs = (("我们的解码", ours, "C2"), ("mbelib 原样移植", mbelib, "C7"))
    ax0 = None
    for i, (title, x, col) in enumerate(sigs):
        t = np.arange(len(x)) / FS
        ax = fig.add_subplot(gs[i], sharex=ax0)
        ax0 = ax0 or ax
        ax.plot(t, x, color=col, lw=0.4)
        ax.set_ylim(-32768, 32767)
        ax.set_title("波形：" + title, loc="left", fontsize=10)
        ax = fig.add_subplot(gs[2 + i], sharex=ax0)
        with np.errstate(divide="ignore"):
            ax.specgram(x, NFFT=256, Fs=FS, noverlap=192, cmap="magma", vmin=-40, vmax=60)
        ax.set_title("语谱图：" + title, loc="left", fontsize=10)
        ax.set_ylabel("Hz")
    ft = np.arange(len(b)) * 0.02
    ax = fig.add_subplot(gs[4], sharex=ax0)
    ax.plot(ft, [T.AmbeW0table[x] * FS if x < 120 else np.nan for x in b[:, 0]], ".", ms=2, color="C3")
    ax.set_ylim(0, 450)
    ax.set_title("基频 f0（Hz）", loc="left", fontsize=10)
    ax = fig.add_subplot(gs[5], sharex=ax0)
    ax.plot(ft, b[:, 2], lw=0.6, color="C4", label="增益 b2")
    ax.plot(ft, b[:, 1], lw=0.6, color="C5", label="清浊音 b1（0 = 全浊）")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("增益和清浊音索引", loc="left", fontsize=10)
    ax = fig.add_subplot(gs[6], sharex=ax0)
    ax.bar(ft, errs, width=0.02, color="C1")
    ax.set_title("每帧 Golay 纠错位数（C0 + C1，> 3 时重复上一帧）  帧类型 %s"
                 % {k: kinds.count(k) for k in sorted(set(kinds))}, loc="left", fontsize=10)
    ax.set_xlabel("秒")
    png = os.path.join(OUT, stem + ".png")
    fig.savefig(png, dpi=80, bbox_inches="tight")
    return png


def main():
    os.makedirs(OUT, exist_ok=True)
    for path in sys.argv[1:]:
        stem = os.path.splitext(os.path.basename(path))[0]
        ours, mbelib, b, errs, kinds = decode(path)
        write_wav(os.path.join(OUT, stem + "_ours.wav"), ours)
        write_wav(os.path.join(OUT, stem + "_mbelib.wav"), mbelib)
        png = plot(stem, ours, mbelib, b, errs, kinds)
        print("%s: %d frames, %.1f s, Golay corrections %d (frames with any %d), kinds %s -> %s"
              % (stem, len(b), len(b) * 0.02, errs.sum(), np.sum(errs > 0),
                 {k: kinds.count(k) for k in sorted(set(kinds))}, os.path.relpath(png)))


if __name__ == "__main__":
    main()
