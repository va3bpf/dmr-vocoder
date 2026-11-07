"""The Tytera MD-380 firmware's DVSI AMBE+2 vocoder (md380tools' md380-emu under qemu in WSL; tools/build_md380emu.sh)
as an encoder / decoder for the tests: a DVSI reference, far faster than a radio
(25 s of speech: encode 1 s, decode 0.6 s).

Frames are 49-bit lists in our d49 order (ambe_frame.decode_frame), which is the order of the .amb files md380-emu
reads and writes (DSD's: 8 bytes a frame, a status byte, the bits MSB first, the last bit as the LSB of byte 8).
"""

import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "tools", "md380emu.sh")


def _wsl(path):
    p = os.path.abspath(path).replace("\\", "/")
    return "/mnt/" + p[0].lower() + p[2:]


def _run(verb, src, dst):
    r = subprocess.run(["wsl", "-u", "root", "--", "bash", _wsl(SCRIPT), verb, _wsl(src), _wsl(dst)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("md380-emu %s: %s" % (verb, r.stderr.strip()))


def _tmpdir():
    d = os.path.join(HERE, "..", "out", "tmp")
    os.makedirs(d, exist_ok=True)
    return tempfile.mkdtemp(dir=d)


def encode(x):
    """PCM (8 kHz) -> list of 49-bit lists, one a 160 samples."""
    d = _tmpdir()
    src, dst = os.path.join(d, "in.raw"), os.path.join(d, "out.amb")
    np.clip(np.round(np.asarray(x, float)), -32768, 32767).astype("<i2").tofile(src)
    _run("enc", src, dst)
    data = open(dst, "rb").read()[4:]
    frames = []
    for k in range(0, len(data) - 7, 8):
        b = data[k + 1:k + 8]
        bits = [(b[i] >> (7 - j)) & 1 for i in range(6) for j in range(8)] + [b[6] & 1]
        frames.append(bits)
    return frames


def decode(frames49):
    """list of 49-bit lists -> PCM (float)."""
    d = _tmpdir()
    src, dst = os.path.join(d, "in.amb"), os.path.join(d, "out.raw")
    out = bytearray(b".amb")
    for bits in frames49:
        packed = [0] + [sum(bits[8 * i + j] << (7 - j) for j in range(8)) for i in range(6)] + [bits[48]]
        out += bytes(packed)
    open(src, "wb").write(out)
    _run("dec", src, dst)
    return np.fromfile(dst, "<i2").astype(float)
