"""Our vocoder on the DM-32, over the programming cable (the radio's debug commands, radio_link.py):

  python vocoder_bench.py time           milliseconds a frame: decoder, encoder and the encoder's steps; the cost of
                                         soft-float and integer operations (CPU clock 168 MHz)
  python vocoder_bench.py exact [frames] bit-exact check: the radio's encoder and decoder against bin/ambe_cli.exe
                                         (the same C code on the PC), frame by frame on out/zh6_input.wav
                                         (zh6_input.py; default 200 frames from frame 100)

The radio: OpenDMR firmware built in ../src (its build/opendmr.elf gives the addresses: the ELF must be the
image on the radio), idle on an FM channel, not in hotspot mode, cable in. On a DMR channel the radio's own audio
uses the vocoder at the same time and the results are wrong.
"""

import struct
import sys

import numpy as np

import c_eval
import zh6_input
from radio_link import Link, symbols

MHZ = 168


def to_bytes(bits72):
    return bytes(int("".join(map(str, bits72[i:i + 8])), 2) for i in range(0, 72, 8))


def our_vocoder(names=()):
    """The link, with our vocoder in its RAM (loaded at the first DMR call: loaded now if not yet)."""
    link = Link(names=("vocoder_sync",) + tuple(names))
    link.call(link.s["vocoder_sync"])
    return link


def time_it():
    link = our_vocoder(("vocoder_bench",))
    fn = link.s["vocoder_bench"]
    for what, n, label in ((0, 200, "float multiply"), (1, 200, "float add"), (2, 200, "int multiply-add")):
        ms = link.big_stack_call(fn, what << 16 | n, 0)
        print("%-17s %6.1f cycles (%d ms for %d000)" % (label, ms * MHZ * 1000 / (n * 1000), ms, n), flush=True)
    x = zh6_input.load()
    data = b"".join(to_bytes(f) for f in c_eval.c_encode(x[:160 * 300])[100:300])  # speech, 200 frames
    link._write_long(link.scratch, data)
    ms = link.big_stack_call(fn, 3 << 16 | 200, link.scratch)
    print("our decoder       %6.2f ms a frame (%d ms for 200; the stock one runs 3 a 60 ms burst)" % (ms / 200, ms))
    link._write_long(link.scratch, x[160 * 150:160 * 154].astype("<i2").tobytes())  # 4 frames of speech
    prof = symbols(("ambe_prof",))["ambe_prof"]
    link.write(prof, bytes(96))  # ambe_prof[AMBE_PROF_N], all the PROF steps
    ms = link.big_stack_call(fn, 4 << 16 | 100, link.scratch)
    print("our encoder       %6.2f ms a frame (%d ms for 100; the stock one about 13)" % (ms / 100, ms))
    p = struct.unpack("<24I", link.read(prof, 96))
    names = ("spectrum", "pitch", "harmonics", "quantize", " track (in pitch)", " harmonic fits (in pitch)",
             " E(P) (in track)", " weighted (in quantize)", " FFT (in spectrum)", "  recon (in weighted)",
             "  smallest (in weighted)", "  e24 (in weighted)", " decoder mirror (in quantize)")
    for name, c in zip(names, p):
        print("  %-30s %6.2f ms a frame" % (name, c / 84000 / 100))
    print("  harmonic fits a frame %.1f, harmonics a fit %.1f; fit_error sums + 64-bit division %.2f ms a frame"
          % (p[16] / 100, p[17] / max(p[16], 1), p[19] / 84000 / 100))


def exact(nframes=200, start=100):
    """vocoderEncodeFrame / vocoderDecodeFrame called one frame at a time on the radio, against ambe_cli."""
    link = our_vocoder(("vocoderEncodeFrame", "vocoderDecodeFrame", "encoder_ready", "decoder_ready"))
    pcm_in, out = link.scratch, link.scratch + 0x200
    x = zh6_input.load()[160 * start:160 * (start + nframes)]

    link.write(link.s["encoder_ready"], b"\x00")  # a fresh encoder, as ambe_cli's
    radio = []
    for k in range(nframes):
        link.write(pcm_in, x[160 * k:160 * (k + 1)].astype("<i2").tobytes())
        link.big_stack_call(link.s["vocoderEncodeFrame"], pcm_in, out)
        radio.append(link.read(out, 9))
    radio = radio[1:]  # the encoder lags a frame (AMBE_ENC_DELAY); ambe_cli drops that first frame
    pc = [to_bytes(f) for f in c_eval.c_encode(x)]
    n = min(len(radio), len(pc))
    bad = [k for k in range(n) if radio[k] != pc[k]]
    print("encoder: %d frames, %d identical%s" % (n, n - len(bad), ", first different %d" % bad[0] if bad else ""),
          flush=True)

    link.write(link.s["decoder_ready"], b"\x00")
    y = []
    for f in pc:
        link.write(pcm_in, f)
        link.big_stack_call(link.s["vocoderDecodeFrame"], pcm_in, out)
        y.append(np.frombuffer(link.read(out, 320), "<i2"))
    y = np.concatenate(y).astype(float)
    ref = c_eval.c_decode([[(b >> (7 - j)) & 1 for b in f for j in range(8)] for f in pc])
    n = min(len(y), len(ref))
    d = y[:n] - ref[:n]
    print("decoder: %d samples, %d different, max |diff| %.0f" % (n, np.count_nonzero(d), np.max(np.abs(d))))
    return not bad and not np.any(d)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    what = sys.argv[1] if len(sys.argv) > 1 else "time"
    if what == "time":
        time_it()
    elif what == "exact":
        sys.exit(0 if exact(*map(int, sys.argv[2:3])) else 1)
    else:
        sys.exit(__doc__)
