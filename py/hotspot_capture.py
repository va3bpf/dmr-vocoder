"""Record DMR voice from the air through the DM-32 in hotspot mode, as AMBE frames for our decoder.

  python hotspot_capture.py [--port COM6] [--freq 433450000] [--cc 1] [--seconds 120] [name]

The radio (our firmware, src/, hotspot mode, cable on --port) is set up like MMDVMHost would: DMR, simplex on
--freq, colour code --cc. Talk from another DMR radio on that frequency and colour code. Every transmission (voice
header to terminator, or a pause over 1 s) is written to vocoder/data/capture/<name>_<n>.bit: 9 bytes (72 bits) per
20 ms frame, the format of DVSI's .bit files, so decode.py / ambe_frame.py read both.

MMDVM DMR frame from the radio: E0 len 18|1A ctrl 33-byte burst rssi(2). ctrl 0x40|type is a data burst (0x41
voice header, 0x42 terminator), otherwise a voice burst (0x20 sync, 1..5 sequence). A voice burst carries
bits 0..107 and 156..263 of AMBE (3 frames x 72 bits); bits 108..155 are the sync or the embedded signalling.
"""

import argparse
import os
import struct
import time

import serial

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "capture")


def frame(cmd, payload=b""):
    return bytes([0xE0, 3 + len(payload), cmd]) + payload


class Link:
    def __init__(self, port):
        self.p = serial.Serial(port, 115200, timeout=0)
        self.buf = b""

    def write(self, data):
        self.p.write(data)

    def frames(self):
        self.buf += self.p.read(4096)
        out = []
        while True:
            i = self.buf.find(b"\xe0")
            if i < 0:
                self.buf = b""
                break
            self.buf = self.buf[i:]
            if len(self.buf) < 2 or len(self.buf) < self.buf[1]:
                break
            out.append(self.buf[:self.buf[1]])
            self.buf = self.buf[self.buf[1]:]
        return out


def burst_ambe(burst33):
    """33-byte DMR burst -> 27 bytes of AMBE (3 frames x 72 bits), MSB first."""
    bits = [(b >> (7 - i)) & 1 for b in burst33 for i in range(8)]
    a = bits[0:108] + bits[156:264]
    return bytes(sum(a[k + j] << (7 - j) for j in range(8)) for k in range(0, 216, 8))


def setup(link, freq, cc):
    link.write(frame(0x00))  # get version
    time.sleep(2)
    link.frames()
    link.write(frame(0x00))
    time.sleep(0.3)
    # set config: no inversion, DMR only, tx delay, state idle, rx / cw levels, colour code, dmr delay, other levels
    link.write(frame(0x02, bytes([0x00, 0x02, 10, 0, 128, 128, cc, 0, 128, 128, 128, 128, 128, 128, 128])))
    time.sleep(0.2)
    link.write(frame(0x04, bytes([0]) + struct.pack("<II", freq, freq) + bytes([100])))  # rx / tx frequency, power
    time.sleep(0.2)
    link.write(frame(0x03, bytes([2])))  # mode DMR
    time.sleep(0.2)
    link.frames()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?", default="zh")
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--freq", type=int, default=433450000)
    ap.add_argument("--cc", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=120)
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    link = Link(a.port)
    setup(link, a.freq, a.cc)
    print("listening on %.4f MHz, colour code %d, for %d s; talk from the other radio" % (a.freq / 1e6, a.cc, a.seconds),
          flush=True)

    take, cur, last = 0, [], 0.0

    def save():
        nonlocal take, cur
        if cur:
            take += 1
            path = os.path.join(OUT, "%s_%d.bit" % (a.name, take))
            with open(path, "wb") as f:
                for b in cur:
                    f.write(b)
            print("  saved %s: %d frames, %.1f s" % (os.path.relpath(path), len(cur) * 3, len(cur) * 0.06), flush=True)
        cur = []

    end = time.time() + a.seconds
    next_status = 0.0
    while time.time() < end:
        now = time.time()
        if now >= next_status:
            link.write(frame(0x01))  # status poll, as MMDVMHost does
            next_status = now + 0.25
        if cur and now - last > 1.0:
            save()
        for f in link.frames():
            if f[2] not in (0x18, 0x1A) or len(f) < 4 + 33:
                if f[2] == 0x7F:
                    print("NAK", f.hex(), flush=True)
                continue
            ctrl, burst = f[3], f[4:4 + 33]
            if ctrl & 0x40:
                kind = ctrl & 0x0F
                if kind == 0x01:
                    print("voice header", flush=True)
                elif kind == 0x02:
                    save()
                    print("terminator", flush=True)
                continue
            cur.append(burst_ambe(burst))
            last = now
        time.sleep(0.005)
    save()


if __name__ == "__main__":
    main()
