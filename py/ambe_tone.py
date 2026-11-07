"""AMBE+2 tone frames (TIA-102.BABA-1 tone frame format, also used by DMR and NXDN).

A tone frame has the first 6 of the 49 bits set. It carries an 8-bit tone index four times and a 7-bit log
amplitude:
  bits 0..5    111111
  bits 6..11   amplitude bits 6..1
  bits 12..19  index (copy 1)
  bits 20..27  index (copy 2)
  bits 28..35  index (copy 3)
  bits 36..43  index (copy 4)
  bit  44      amplitude bit 0
  bits 45..48  0
Index 5..122: one tone at 31.25 * index Hz; 128..163: the dual tones below (DTMF, KNOX, call progress);
255: silence. Amplitude 127 is a full-scale tone (32767 peak), each step 0.711 dB less.

Published vector (NXDN TS 1-A 7.2.1): index 33 = 1031.25 Hz, 72 bits CEA8FE83ACC458200A.
"""

import math

import numpy as np

# Dual tones: index -> (higher, lower) Hz
DUAL = {
    128: (1336, 941), 129: (1209, 697), 130: (1336, 697), 131: (1477, 697),  # DTMF 0-3
    132: (1209, 770), 133: (1336, 770), 134: (1477, 770), 135: (1209, 852),  # DTMF 4-7
    136: (1336, 852), 137: (1477, 852), 138: (1633, 697), 139: (1633, 770),  # DTMF 8, 9, A, B
    140: (1633, 852), 141: (1633, 941), 142: (1209, 941), 143: (1477, 941),  # DTMF C, D, *, #
    144: (1162, 820), 145: (1052, 606), 146: (1162, 606), 147: (1279, 606),  # KNOX 0-3
    148: (1052, 672), 149: (1162, 672), 150: (1279, 672), 151: (1052, 743),  # KNOX 4-7
    152: (1162, 743), 153: (1279, 743), 154: (1430, 606), 155: (1430, 672),  # KNOX 8, 9, A, B
    156: (1430, 743), 157: (1430, 820), 158: (1052, 820), 159: (1279, 820),  # KNOX C, D, *, #
    160: (440, 350), 161: (480, 440), 162: (620, 480), 163: (490, 350),      # call progress
}
DB_PER_STEP = 0.711


def is_tone(d49):
    return all(d49[0:6])


def _field(d, lo, n):
    v = 0
    for i in range(lo, lo + n):
        v = (v << 1) | d[i]
    return v


def parse(d49):
    """-> (tone index by majority of the four copies, 7-bit amplitude)."""
    amp = (_field(d49, 6, 6) << 1) | d49[44]
    copies = [_field(d49, 12 + 8 * k, 8) for k in range(4)]
    idx = max(copies, key=copies.count)
    return idx, amp


def build(idx, amp):
    """49 bits of a tone frame."""
    d = [1] * 6 + [(amp >> (6 - i)) & 1 for i in range(6)]
    for _ in range(4):
        d += [(idx >> (7 - i)) & 1 for i in range(8)]
    d += [amp & 1, 0, 0, 0, 0]
    return d


def frequencies(idx):
    """Tone index -> list of frequencies in Hz ([] for silence), or None if the index is not a tone."""
    if idx == 255:
        return []
    if 5 <= idx <= 122:
        return [31.25 * idx]
    if idx in DUAL:
        return list(DUAL[idx])
    return None


def peak(amp):
    """Peak amplitude of each component, 16-bit PCM."""
    return 32767 * 10 ** (DB_PER_STEP * (amp - 127) / 20)


class ToneSynth:
    """Tones with continuous phase; the amplitude ramps over a frame from the previous frame's."""

    def __init__(self):
        self.phase = {}  # frequency -> phase at the start of the next frame
        self.level = {}  # frequency -> amplitude at the end of the last frame

    def frame(self, freqs, a, N=160):
        n = np.arange(N)
        out = np.zeros(N)
        phase, level = {}, {}
        for f in set(freqs) | set(self.level):
            a0 = self.level.get(f, 0.0)
            a1 = a if f in freqs else 0.0
            w = 2 * math.pi * f / 8000
            p = self.phase.get(f, 0.0)
            out += (a0 + (a1 - a0) * n / N) * np.cos(p + w * n)
            if a1:
                phase[f] = (p + w * N) % (2 * math.pi)
                level[f] = a1
        self.phase, self.level = phase, level
        return out


if __name__ == "__main__":
    from ambe_frame import decode_frame, encode_frame

    bits = [(x >> (7 - i)) & 1 for x in bytes.fromhex("CEA8FE83ACC458200A") for i in range(8)]
    d, e0, e1 = decode_frame(bits)
    assert is_tone(d) and e0 == e1 == 0
    idx, amp = parse(d)
    assert build(idx, amp) == d and encode_frame(build(idx, amp)) == bits
    print("published tone: index %d = %s Hz, amplitude %d (%.1f dBFS)"
          % (idx, frequencies(idx), amp, DB_PER_STEP * (amp - 127)))
