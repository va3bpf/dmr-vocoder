"""The C vocoder (bin/ambe_cli.exe, c/ambe, the embedded version) against the Python one and the double reference
(bin/ambe_ref.exe, c/ref), on the md380_eval.py inputs:

  encoder: the frames' b0..b8 from C and from ambe_encoder.encode_pcm (how many frames agree, field by field),
           and PESQ / STOI of both through md380's decoder;
  decoder: PESQ / STOI and RMS of md380's bits decoded by c/ambe, c/ref and ambe_decoder.Decoder.

  python c_eval.py [enc] [dec]     (both by default)
"""

import json
import os
import subprocess
import sys

import numpy as np

import ambe_encoder
import md380emu
from ambe_decoder import unpack_b
from ambe_frame import decode_frame, encode_frame
from compare import OUT
from md380_eval import inputs, ours_decode, scores

BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin")
CLI = os.path.join(BIN, "ambe_cli.exe")
REF = os.path.join(BIN, "ambe_ref.exe")
TMP = os.path.join(OUT, "tmp")
os.makedirs(TMP, exist_ok=True)


def c_encode(x):
    src, dst = os.path.join(TMP, "c_in.raw"), os.path.join(TMP, "c_out.bit")
    np.clip(np.round(x), -32768, 32767).astype("<i2").tofile(src)
    subprocess.run([CLI, "enc", src, dst], check=True)
    d = open(dst, "rb").read()
    return [[(d[i + j // 8] >> (7 - j % 8)) & 1 for j in range(72)] for i in range(0, len(d), 9)]


def c_decode(frames72, exe=CLI):
    src, dst = os.path.join(TMP, "c_in.bit"), os.path.join(TMP, "c_out.raw")
    open(src, "wb").write(b"".join(bytes(int("".join(map(str, f[i:i + 8])), 2) for i in range(0, 72, 8))
                                   for f in frames72))
    subprocess.run([exe, "dec", src, dst], check=True)
    return np.fromfile(dst, "<i2").astype(float)


def rms(y):
    return float(np.sqrt(np.mean(np.asarray(y, float) ** 2)))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    what = sys.argv[1:] or ["enc", "dec"]
    for name, x in inputs().items():
        line = "%-5s" % name
        if "enc" in what:
            fc = c_encode(x)
            fp = ambe_encoder.encode_pcm(x)[0]
            bc = np.array([unpack_b(decode_frame(f)[0]) for f in fc])
            bp = np.array([unpack_b(decode_frame(f)[0]) for f in fp])
            n = min(len(bc), len(bp))
            same = np.mean(np.all(bc[:n] == bp[:n], axis=1))
            fields = " ".join("%.0f" % (100 * np.mean(bc[:n, i] == bp[:n, i])) for i in range(9))
            ec = scores(x, md380emu.decode([decode_frame(f)[0] for f in fc]))
            ep = scores(x, md380emu.decode([decode_frame(f)[0] for f in fp]))
            line += " enc: frames %d/%d same %.1f%%, b0..b8 %s%% | PESQ/STOI C %.3f/%.3f py %.3f/%.3f" % (
                len(bc), len(bp), 100 * same, fields, *ec, *ep)
        if "dec" in what:
            m = [encode_frame(f) for f in json.load(open(os.path.join(OUT, "md380_%s.json" % name)))]
            yc, yr, yp = c_decode(m), c_decode(m, REF), ours_decode([decode_frame(f)[0] for f in m])
            line += " dec: PESQ/STOI/RMS C %.3f/%.3f/%.0f ref %.3f/%.3f/%.0f py %.3f/%.3f/%.0f" % (
                *scores(x, yc), rms(yc), *scores(x, yr), rms(yr), *scores(x, yp), rms(yp))
        print(line, flush=True)


if __name__ == "__main__":
    main()
