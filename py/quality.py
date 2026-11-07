"""Perceptual measures of decoded speech against the original: PESQ (ITU-T P.862 narrow band, MOS-LQO 1..4.55,
vocoder/bin/pesq_nb.exe, built by c/build.sh) and STOI (intelligibility 0..1, pystoi)."""

import os
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PESQ_EXE = os.path.join(HERE, "..", "bin", "pesq_nb.exe")


def pesq(ref, deg, fs=8000):
    """MOS-LQO of deg against ref (16-bit PCM arrays; PESQ finds the delay itself)."""
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for name, x in (("ref", ref), ("deg", deg)):
            p = os.path.join(d, name + ".raw")
            np.clip(np.asarray(x), -32768, 32767).astype("<i2").tofile(p)
            paths.append(p)
        r = subprocess.run([os.path.abspath(PESQ_EXE)] + paths + [str(fs)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("pesq_nb: " + r.stderr.strip())
    return float(r.stdout.split()[0])


def stoi(ref, deg, lag=0, fs=8000):
    """STOI of deg against ref, deg shifted by lag samples."""
    from pystoi import stoi as _stoi

    n = min(len(ref), len(deg) - lag)
    return _stoi(np.asarray(ref[:n], float), np.asarray(deg[lag:lag + n], float), fs)
