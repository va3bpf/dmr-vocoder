"""Transmit speech coded by our encoder through the DM-32 in hotspot mode, to be heard on another DMR radio.

  python hotspot_tx.py file ... [--bpf 300 3400] [--gap 0.4] [--port COM6] [--freq 433450000] [--cc 1] [--tg 9]
                       [--src 4600001] [--save out.bit]

Several .wav (any rate, resampled to 8 kHz) or .pcm (8 kHz 16-bit) files are joined with --gap seconds of silence,
band-passed (--bpf, 4th order Butterworth, zero phase; --bpf 0 0 for none), brought to an active speech level of
about 3000 RMS and encoded with ambe_encoder.encode_pcm; a .bit file (9 bytes per frame, as DVSI's and
hotspot_capture.py's) is sent as it is. The radio (our firmware, hotspot mode,
cable on --port) is set up as by hotspot_capture.py, then gets the call as MMDVMHost would send it from the
network: a voice LC header (group call to --tg from --src), voice bursts A..F every 60 ms, a terminator with LC.

The header and terminator are coded as services/hotspot.c codes them (DMRFullLC_encode: RS(12,9) with the CRC
mask, BPTC(196,96)), and a voice burst carries the AMBE where storeNetFrame() takes it: bits 0..107 and 156..263.
"""

import argparse
import math
import os
import time

import numpy as np

from hotspot_capture import Link, frame, setup

SILENCE = bytes.fromhex("B9E881526173002A6B")  # AMBE+2 silence frame (MMDVMHost DMR_SILENCE_DATA)
BS_VOICE_SYNC = bytes.fromhex("755FD7DF75F7")    # base station sourced voice sync, the middle 48 bits of burst A
VOICE_LC_HEADER_MASK = 0x96
TERMINATOR_MASK = 0x99
COPY_RANGES = ((4, 11), (16, 26), (31, 41), (46, 56), (61, 71), (76, 86), (91, 101), (106, 116), (121, 131))

_EXP = [0] * 512
_LOG = [0] * 256
_v = 1
for _i in range(255):
    _EXP[_i] = _v
    _LOG[_v] = _i
    _v <<= 1
    if _v & 0x100:
        _v ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a, b):
    return 0 if a == 0 or b == 0 else _EXP[(_LOG[a] + _LOG[b]) % 255]


def rs129(data9):
    """hotspot.c ReedSolomonDMREncode: 3 parity bytes, as out[0..2] (the LC gets out[2], out[1], out[0])."""
    f = (64, 56, 14)
    out = [0, 0, 0]
    for b in data9:
        t = b ^ out[2]
        out[2] = out[1] ^ _mul(f[2], t)
        out[1] = out[0] ^ _mul(f[1], t)
        out[0] = _mul(f[0], t)
    return out


def _bits(byte):
    return [(byte >> (7 - i)) & 1 for i in range(8)]


def _byte(bits):
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v


def bptc_encode(lc12):
    """hotspot.c BPTCencode: 12 bytes -> the 33-byte burst's BPTC part (sync and slot type left 0)."""
    data = [b for x in lc12 for b in _bits(x)]
    d = [0] * 196
    p = 0
    for a, z in COPY_RANGES:
        for i in range(a, z + 1):
            d[i] = data[p]
            p += 1
    for r in range(9):  # Hamming (15,11,3) rows
        x = d[r * 15 + 1:r * 15 + 16]
        x[11] = x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[5] ^ x[7] ^ x[8]
        x[12] = x[1] ^ x[2] ^ x[3] ^ x[4] ^ x[6] ^ x[8] ^ x[9]
        x[13] = x[2] ^ x[3] ^ x[4] ^ x[5] ^ x[7] ^ x[9] ^ x[10]
        x[14] = x[0] ^ x[1] ^ x[2] ^ x[4] ^ x[6] ^ x[7] ^ x[10]
        d[r * 15 + 1:r * 15 + 16] = x
    for c in range(15):  # Hamming (13,9,3) columns
        h = [d[c + 1 + 15 * j] for j in range(13)]
        h[9] = h[0] ^ h[1] ^ h[3] ^ h[5] ^ h[6]
        h[10] = h[0] ^ h[1] ^ h[2] ^ h[4] ^ h[6] ^ h[7]
        h[11] = h[0] ^ h[1] ^ h[2] ^ h[3] ^ h[5] ^ h[7] ^ h[8]
        h[12] = h[0] ^ h[2] ^ h[4] ^ h[5] ^ h[8]
        for j in range(13):
            d[c + 1 + 15 * j] = h[j]
    raw = [0] * 196
    for i in range(196):
        raw[(i * 181) % 196] = d[i]
    out = [0] * 33
    for i in range(12):
        out[i] = _byte(raw[i * 8:i * 8 + 8])
    b = _byte(raw[96:104])
    out[12] = (out[12] & 0x3F) | (b & 0xC0)
    out[20] = (out[20] & 0xFC) | ((b >> 4) & 0x03)
    for i in range(12):
        out[21 + i] = _byte(raw[100 + i * 8:108 + i * 8])
    return bytes(out)


def lc_burst(tg, src, mask):
    lc = [0x00, 0x00, 0x00, (tg >> 16) & 0xFF, (tg >> 8) & 0xFF, tg & 0xFF, (src >> 16) & 0xFF, (src >> 8) & 0xFF,
          src & 0xFF]  # FLCO 0 (group call), FID 0, service options 0
    p = rs129(lc)
    return bptc_encode(lc + [p[2] ^ mask, p[1] ^ mask, p[0] ^ mask])


def voice_burst(ambe27, middle=bytes(6)):
    """27 AMBE bytes -> 33-byte burst: AMBE bits 0..107, 48 middle bits, AMBE bits 108..215."""
    a = [b for x in ambe27 for b in _bits(x)]
    m = [b for x in middle for b in _bits(x)]
    bits = a[:108] + m + a[108:]
    return bytes(_byte(bits[i:i + 8]) for i in range(0, 264, 8))


def read_audio(path):
    """.wav (resampled to 8 kHz) or .pcm (8 kHz) -> float samples."""
    if path.endswith(".wav"):
        import wave
        from math import gcd

        from scipy.signal import resample_poly

        w = wave.open(path)
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float)
        fs = w.getframerate()
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(axis=1)
        g = gcd(fs, 8000)
        return resample_poly(x, 8000 // g, fs // g)
    return np.fromfile(path, "<i2").astype(float)


def prepare(paths, bpf=(300, 3400), gap=0.4):
    """Join, band-pass and level the inputs -> 8 kHz float samples."""
    parts = []
    for p in paths:
        parts += [read_audio(p), np.zeros(int(gap * 8000))]
    x = np.concatenate(parts)
    if bpf and bpf[0] < bpf[1]:
        from scipy.signal import butter, sosfiltfilt

        x = sosfiltfilt(butter(4, bpf, btype="bandpass", fs=8000, output="sos"), x)
    return x


def load_frames(paths, bpf=(300, 3400), gap=0.4):
    """-> list of 9-byte AMBE frames."""
    if len(paths) == 1 and paths[0].endswith(".bit"):
        d = open(paths[0], "rb").read()
        return [d[i:i + 9] for i in range(0, len(d) - 8, 9)]
    from ambe_encoder import encode_pcm

    x = prepare(paths, bpf, gap)
    # active speech level about 3000 RMS (DVSI's test speech), never clipping
    fe = np.mean(x[:len(x) // 160 * 160].reshape(-1, 160) ** 2, axis=1)
    level = math.sqrt(fe[fe > fe.max() / 1000].mean())
    x = x * min(3000 / level, 30000 / np.abs(x).max())
    frames, _ = encode_pcm(x)
    out = []
    for f in frames:
        out.append(bytes(_byte(f[i:i + 8]) for i in range(0, 72, 8)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="+")
    ap.add_argument("--bpf", type=float, nargs=2, default=(300, 3400))
    ap.add_argument("--gap", type=float, default=0.4)
    ap.add_argument("--save", help="also write the frames to this .bit file")
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--freq", type=int, default=433450000)
    ap.add_argument("--cc", type=int, default=1)
    ap.add_argument("--tg", type=int, default=9)
    ap.add_argument("--src", type=int, default=4600001)
    a = ap.parse_args()

    frames = load_frames(a.input, tuple(a.bpf), a.gap)
    if a.save:
        open(a.save, "wb").write(b"".join(frames))
    frames += [SILENCE] * ((-len(frames)) % 3)
    bursts = [voice_burst(b"".join(frames[i:i + 3]), BS_VOICE_SYNC if (i // 3) % 6 == 0 else bytes(6))
              for i in range(0, len(frames), 3)]
    print("%d file(s): %d frames, %.1f s, %d bursts" % (len(a.input), len(frames), len(frames) * 0.02, len(bursts)),
          flush=True)

    link = Link(a.port)
    setup(link, a.freq, a.cc)
    out = [frame(0x1A, bytes([0x41]) + lc_burst(a.tg, a.src, VOICE_LC_HEADER_MASK))] * 3
    out += [frame(0x1A, bytes([0x20 if n % 6 == 0 else n % 6]) + b) for n, b in enumerate(bursts)]
    out += [frame(0x1A, bytes([0x42]) + lc_burst(a.tg, a.src, TERMINATOR_MASK))] * 2
    print("sending to TG %d from %d on %.4f MHz, colour code %d" % (a.tg, a.src, a.freq / 1e6, a.cc), flush=True)

    t = time.time() + 0.1
    next_status = 0.0
    naks = 0
    while out or time.time() < t + 3:
        now = time.time()
        if now >= next_status:
            link.write(frame(0x01))
            next_status = now + 0.25
        if out and now >= t:
            link.write(out.pop(0))
            t += 0.06
        for f in link.frames():
            if f[2] == 0x7F:
                naks += 1
                print("NAK", f.hex(), flush=True)
        time.sleep(0.003)
    print("done%s" % (", %d NAK" % naks if naks else ""))


if __name__ == "__main__":
    main()
