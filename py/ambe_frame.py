"""DMR AMBE+2 (3600x2450) frame layer: 72 channel bits <-> 49 vocoder bits.

The 72 bits of a frame are interleaved into four code vectors (dsd include/dmr_const.h):
  C0: 24 bits, Golay(24,12)  - fr[0][0] is the overall parity, fr[0][1..23] a Golay(23,12) block
  C1: 23 bits, Golay(23,12), XOR-ed with a pseudo-random sequence seeded by the 12 data bits of C0
  C2: 11 bits, no protection
  C3: 14 bits, no protection
12 + 12 + 11 + 14 = 49 vocoder bits (mbelib ambe3600x2450.c, mbe_eccAmbe3600x2450Data).

A DMR voice burst carries 3 frames = 216 bits = 27 bytes (AMBE_AUDIO_LENGTH in the firmware, the hotspot
buffer and MMDVM), bits MSB first.
"""

from itertools import combinations

# DMR AMBE interleave schedule: channel bit 2i -> fr[W[i]][X[i]], channel bit 2i+1 -> fr[Y[i]][Z[i]]
W = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2)
X = (23, 10, 22, 9, 21, 8, 20, 7, 19, 6, 18, 5, 17, 4, 16, 3, 15, 2, 14, 1, 13, 0, 12, 10,
     11, 9, 10, 8, 9, 7, 8, 6, 7, 5, 6, 4)
Y = (0, 2, 0, 2, 0, 2, 0, 2, 0, 3, 0, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3)
Z = (5, 3, 4, 2, 3, 1, 2, 0, 1, 13, 0, 12, 22, 11, 21, 10, 20, 9, 19, 8, 18, 7, 17, 6,
     16, 5, 15, 4, 14, 3, 13, 2, 12, 1, 11, 0)

# Golay(23,12) parity of each data bit, data MSB first (mbelib ecc_const.h golayGenerator)
GOLAY_GEN = (0x63a, 0x31d, 0x7b4, 0x3da, 0x1ed, 0x6cc, 0x366, 0x1b3, 0x6e3, 0x54b, 0x49f, 0x475)


def golay23_parity(data):
    p = 0
    for i in range(12):
        if data & (0x800 >> i):
            p ^= GOLAY_GEN[i]
    return p


def golay23_encode(data):
    """12 data bits -> 23-bit block, data in bits 22..11, parity in bits 10..0."""
    return (data << 11) | golay23_parity(data)


def _syndrome(block):
    return golay23_parity(block >> 11) ^ (block & 0x7FF)


def _build_golay23_table():
    # The code is perfect: the 2048 patterns of weight <= 3 have exactly the 2048 syndromes
    table = {0: 0}
    for w in (1, 2, 3):
        for pos in combinations(range(23), w):
            e = 0
            for p in pos:
                e |= 1 << p
            table[_syndrome(e)] = e
    assert len(table) == 2048
    return table


_GOLAY23_ERR = _build_golay23_table()


def golay23_decode(block):
    """23-bit block -> (12 data bits, corrected bit count)."""
    e = _GOLAY23_ERR[_syndrome(block)]
    return (block ^ e) >> 11, bin(e).count("1")


def _pn(seed12):
    """The 23 bits XOR-ed onto C1, for fr[1][22] down to fr[1][0] (mbe_demodulateAmbe3600x2450Data)."""
    pr = 16 * seed12
    out = []
    for _ in range(23):
        pr = (173 * pr + 13849) & 0xFFFF
        out.append(pr >> 15)
    return out


def deinterleave(bits72):
    fr = [[0] * 24 for _ in range(4)]
    for i in range(36):
        fr[W[i]][X[i]] = bits72[2 * i]
        fr[Y[i]][Z[i]] = bits72[2 * i + 1]
    return fr


def interleave(fr):
    bits = [0] * 72
    for i in range(36):
        bits[2 * i] = fr[W[i]][X[i]]
        bits[2 * i + 1] = fr[Y[i]][Z[i]]
    return bits


def _vec_to_int(v, hi, lo):
    """fr[c][hi] .. fr[c][lo] as an integer, fr[c][hi] the MSB."""
    n = 0
    for j in range(hi, lo - 1, -1):
        n = (n << 1) | v[j]
    return n


def _int_to_vec(v, hi, lo, n):
    for j in range(lo, hi + 1):
        v[j] = (n >> (j - lo)) & 1


def decode_frame(bits72):
    """72 channel bits -> (49 vocoder bits, errors corrected in C0, errors corrected in C1)."""
    fr = deinterleave(bits72)
    c0, e0 = golay23_decode(_vec_to_int(fr[0], 23, 1))
    pn = _pn(c0)
    for k, j in enumerate(range(22, -1, -1)):
        fr[1][j] ^= pn[k]
    c1, e1 = golay23_decode(_vec_to_int(fr[1], 22, 0))
    d = [(c0 >> (11 - i)) & 1 for i in range(12)]
    d += [(c1 >> (11 - i)) & 1 for i in range(12)]
    d += [fr[2][j] for j in range(10, -1, -1)]
    d += [fr[3][j] for j in range(13, -1, -1)]
    return d, e0, e1


def encode_frame(d49):
    """49 vocoder bits -> 72 channel bits."""
    fr = [[0] * 24 for _ in range(4)]
    c0 = _bits_to_int(d49[0:12])
    c1 = _bits_to_int(d49[12:24])
    _int_to_vec(fr[0], 23, 1, golay23_encode(c0))
    fr[0][0] = sum(fr[0][1:24]) & 1  # overall parity, even (not checked by mbelib; to be confirmed on air)
    _int_to_vec(fr[1], 22, 0, golay23_encode(c1))
    pn = _pn(c0)
    for k, j in enumerate(range(22, -1, -1)):
        fr[1][j] ^= pn[k]
    for i, j in enumerate(range(10, -1, -1)):
        fr[2][j] = d49[24 + i]
    for i, j in enumerate(range(13, -1, -1)):
        fr[3][j] = d49[35 + i]
    return interleave(fr)


def _bits_to_int(bits):
    n = 0
    for b in bits:
        n = (n << 1) | b
    return n


def burst_to_frames(data27):
    """27 bytes (one DMR voice burst) -> 3 lists of 72 channel bits."""
    bits = [(byte >> (7 - i)) & 1 for byte in data27 for i in range(8)]
    return [bits[0:72], bits[72:144], bits[144:216]]


def frames_to_burst(frames):
    bits = frames[0] + frames[1] + frames[2]
    return bytes(_bits_to_int(bits[i:i + 8]) for i in range(0, 216, 8))


if __name__ == "__main__":
    import random

    rnd = random.Random(1)
    # The interleave schedule hits every bit of C0 (24), C1 (23), C2 (11), C3 (14) exactly once
    cells = sorted(set(zip(W, X)) | set(zip(Y, Z)))
    assert len(cells) == 72 and all(c < 4 for c, _ in cells)
    for _ in range(2000):
        d = [rnd.getrandbits(1) for _ in range(49)]
        bits = encode_frame(d)
        assert decode_frame(bits) == (d, 0, 0)
        # Up to 3 errors in C0 and up to 3 in C1 are corrected; C0 errors must not upset the C1 descrambling
        fr = deinterleave(bits)
        n0, n1 = rnd.randint(0, 3), rnd.randint(0, 3)
        for j in rnd.sample(range(1, 24), n0):
            fr[0][j] ^= 1
        for j in rnd.sample(range(0, 23), n1):
            fr[1][j] ^= 1
        assert decode_frame(interleave(fr)) == (d, n0, n1)
        burst = frames_to_burst([bits, bits, bits])
        assert len(burst) == 27 and burst_to_frames(burst) == [bits, bits, bits]
    print("ambe_frame: ok")
