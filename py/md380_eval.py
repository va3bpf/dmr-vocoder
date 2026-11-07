"""Score our encoder and decoder against the DVSI vocoder of md380-emu (md380emu.py), which matches the radio's stock
codec (on the six Chinese sentences md380 2.967 PESQ, the radio 2.967): fast, no radio needed.

  python md380_eval.py [module.NAME=value ...]       e.g. ambe_analysis.VOICED_ERR=0.5 "ambe_decoder.DEFAULTS={...}"

For out/zh6_input.wav and DVSI's dam and clean: PESQ / STOI against the input of
  encoder: ours -> md380 decoder         (what a commercial radio plays from us)    against md380 -> md380
  decoder: md380 -> ours                 (what we play from a commercial radio)     against md380 -> md380
and the frame-to-frame pitch steps of the voiced frames, ours against md380's. md380's results are cached in
out/md380_eval_ref.json.
"""

import importlib
import json
import os
import sys

import numpy as np

import md380emu
import mbe_tables as T
import quality
from ambe_decoder import Decoder, to_pcm16, unpack_b
from ambe_frame import encode_frame
from compare import DVSI, OUT, read_pcm
from evaluate import spectral_lag
import zh6_input

REF = os.path.join(OUT, "md380_eval_ref.json")
FILES = os.environ.get("FILES", "zh6 dam clean").split()


def inputs():
    out = {}
    for name in FILES:
        if name == "zh6":
            x = zh6_input.load()
        else:
            x = read_pcm(os.path.join(DVSI, name + ".pcm"))
        out[name] = x[:len(x) // 160 * 160]
    return out


def pitch_steps(frames49):
    B = np.array([unpack_b(f) for f in frames49])
    v = (B[:, 0] < 120) & (np.array([np.mean(T.AmbeVuv[b1][:3]) for b1 in B[:, 1]]) > 0.5)
    d = np.abs(np.diff(B[:, 0]))[v[1:] & v[:-1]]
    return [float(np.mean(d == 0)), float(np.mean(d == 1)), float(np.mean(d > 2))]


def scores(x, y):
    return [quality.pesq(x, y), quality.stoi(x, y, spectral_lag(x, y, 0, 800))]


def ours_decode(frames49):
    dec = Decoder()
    return np.concatenate([to_pcm16(dec.decode_frame72(encode_frame(f))[0]).astype(float) for f in frames49])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for arg in sys.argv[1:]:
        name, value = arg.split("=", 1)
        mod, attr = name.rsplit(".", 1)
        setattr(importlib.import_module(mod), attr, eval(value))
    import ambe_encoder

    data = inputs()
    ref = json.load(open(REF)) if os.path.exists(REF) else {}
    for name, x in data.items():
        if name not in ref:
            m = md380emu.encode(x)
            json.dump(m, open(os.path.join(OUT, "md380_%s.json" % name), "w"))
            ref[name] = {"md380": scores(x, md380emu.decode(m)), "steps": pitch_steps(m)}
            json.dump(ref, open(REF, "w"))
    tot_e = tot_d = 0.0
    print(" ".join(sys.argv[1:]) or "defaults")
    for name, x in data.items():
        m = json.load(open(os.path.join(OUT, "md380_%s.json" % name)))
        ours = [ambe_encoder_frame for ambe_encoder_frame in
                (_d49(f) for f in ambe_encoder.encode_pcm(x)[0])]
        enc = scores(x, md380emu.decode(ours))
        dec = scores(x, ours_decode(m))
        r = ref[name]
        tot_e += enc[0] - r["md380"][0]
        tot_d += dec[0] - r["md380"][0]
        st, rst = pitch_steps(ours), r["steps"]
        print("  %-5s md380 %.3f/%.3f | encoder %.3f/%.3f (%+.3f) | decoder %.3f/%.3f (%+.3f) | pitch steps 0/1/>2 "
              "%.0f/%.0f/%.0f%% (md380 %.0f/%.0f/%.0f%%)" % (name, *r["md380"], *enc, enc[0] - r["md380"][0], *dec,
                                                          dec[0] - r["md380"][0], *[100 * v for v in st],
                                                          *[100 * v for v in rst]))
    print("  PESQ against md380, summed: encoder %+.3f decoder %+.3f" % (tot_e, tot_d), flush=True)


def _d49(bits72):
    from ambe_frame import decode_frame

    return decode_frame(bits72)[0]


if __name__ == "__main__":
    main()
