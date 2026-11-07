"""The test speech "zh6": six THCHS-30 sentences (data/thchs30: D4, D6, D7, D8, D11, D12), joined with 0.4 s gaps,
band-passed 300..3400 Hz and levelled to an active speech level of about 3000 RMS, as
hotspot_tx.py prepares speech for the encoder. 58.3 s, 2915 frames.

  python zh6_input.py      write out/zh6_input.wav (8 kHz 16-bit mono)

The scripts that use it (md380_eval.py, vocoder_bench.py, c_eval.py) call load(), which writes the
file first when it is not there.
"""
import math
import os
import wave

import numpy as np

from compare import OUT, write_wav
from hotspot_tx import prepare

HERE = os.path.dirname(os.path.abspath(__file__))
SENTENCES = ("D4", "D6", "D7", "D8", "D11", "D12")
PATH = os.path.join(OUT, "zh6_input.wav")


def make():
    x = prepare([os.path.join(HERE, "..", "data", "thchs30", "%s_750.wav" % s) for s in SENTENCES])
    fe = np.mean(x[:len(x) // 160 * 160].reshape(-1, 160) ** 2, axis=1)
    x = x * min(3000 / math.sqrt(fe[fe > fe.max() / 1000].mean()), 30000 / np.abs(x).max())
    write_wav(PATH, x[:len(x) // 160 * 160])


def load():
    """-> the samples as floats (whole frames)."""
    if not os.path.exists(PATH):
        make()
    w = wave.open(PATH)
    return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float)


if __name__ == "__main__":
    make()
    print("%s: %.1f s" % (PATH, len(load()) / 8000))
