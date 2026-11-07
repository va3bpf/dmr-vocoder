"""The debug commands of our firmware (src/services/debug.c; tools/debug/dm32dbg.py) over the programming cable, for
the radio tests of the vocoder (vocoder_bench.py): read and write the RAM, call a function, and call it on a big
stack. The radio must run the image whose ELF is src/build (the addresses come from it) and must not be in hotspot
mode (the hotspot owns the cable then). While it runs, idle_activity() is called every KEEPALIVE_S as a key press
would, so the auto power off does not cut a long run short.

The vocoder needs far more stack than the cable task that runs the debug calls has (2 KB; it runs on the HR_C7000
task's 8 KB in the firmware), so the calls go through a trampoline (c/stack_tramp.S, uploaded to the start of the free
RAM) onto a stack in the RAM nothing uses between the IRQ stack (g_top_irqstack in the ELF) and the vocoder (0x44400):
the firmware has no heap.
"""

import os
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
ELF = os.path.join(ROOT, "src", "build", "opendmr.elf")
NM = os.path.join(ROOT, "toolchain", "MinGW", "csky-elfabiv2-tools", "bin", "csky-elfabiv2-nm.exe")
# c/stack_tramp.S, assembled (r0 = function, r1 = stack top, r2, r3 = its arguments); uploaded to the free RAM
TRAMP_CODE = bytes.fromhex(
    "2214eedd00200edd01203b6e876f036f0b6c4f6cf17ba36f0ed90120eed9002002143c782514eedd00200edd01202edd02204edd0320"
    "6edd04203b6e876f436e8b6ecf6e4beb0000080c2b6ce57b4ae508006be50010f707a36f6ed904204ed903202ed902200ed90120eed9"
    "002005143c78")
CHUNK = 1280            # bytes a debug write or read (the cable link buffer is 1536)
STACK_TOP = 0x00044380  # just below the vocoder's RAM (0x44400, src/ckcpu.ld)
KEEPALIVE_S = 30.0      # idle_activity() this often: the auto power off must not switch the radio off in a long run


def symbols(names):
    out = subprocess.run([NM, ELF], capture_output=True, text=True).stdout
    table = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] in names:
            table[parts[2]] = int(parts[0], 16)
    missing = set(names) - set(table)
    if missing:
        raise RuntimeError("not in the ELF: %s" % ", ".join(sorted(missing)))
    return table


class Link:
    """The debug commands over the cable and the stack trampoline at the start of the free RAM (self.tramp, the
    scratch buffer after it)."""

    def __init__(self, port="COM6", names=()):
        import serial

        self.p = serial.Serial(port, 115200, timeout=0.5)
        self.last_keepalive = 0.0
        time.sleep(0.2)
        self.s = symbols(tuple(names) + ("g_top_irqstack", "idle_activity"))
        self.tramp = (self.s["g_top_irqstack"] + 0xFF) & ~0xFF
        self.scratch = self.tramp + 0x100
        assert self.scratch + 0x800 <= STACK_TOP - 0x1000, "too little free RAM for the trampoline"
        self.write(self.tramp, TRAMP_CODE)
        if self.read(self.tramp, len(TRAMP_CODE)) != TRAMP_CODE:
            raise RuntimeError("the trampoline did not stick")

    def big_stack_call(self, fn, a0=0, a1=0):
        """fn(a0, a1) on the big stack."""
        return self.call(self.tramp, fn, STACK_TOP, a0, a1)

    def _request(self, data, reply_len, timeout=2.0):
        for _ in range(3):
            self.p.reset_input_buffer()
            self.p.write(data)
            buf = b""
            end = time.time() + timeout
            while time.time() < end:
                buf += self.p.read(max(1, self.p.in_waiting))
                i = buf.find(data[:2])
                if i >= 0 and len(buf) - i >= reply_len:
                    return buf[i:i + reply_len]
            time.sleep(0.2)
        raise RuntimeError("no reply to %r" % data[:2])

    def write(self, addr, data):
        self._request(b"DW" + struct.pack(">IH", addr, len(data)) + data, 2)

    def read(self, addr, n):
        return self._request(b"Dm" + struct.pack(">IH", addr, n), 2 + n)[2:]

    def call(self, addr, *args):
        now = time.time()
        if now - self.last_keepalive > KEEPALIVE_S and addr != self.s["idle_activity"]:
            self.last_keepalive = now
            self.call(self.s["idle_activity"])  # as a key press: the auto power off counts from now
        a = list(args) + [0] * (4 - len(args))
        return struct.unpack(">I", self._request(b"Dx" + struct.pack(">IIIII", addr, *a), 6)[2:])[0]

    def _write_long(self, addr, data):
        for i in range(0, len(data), CHUNK):
            self.write(addr + i, data[i:i + CHUNK])

    def _read_long(self, addr, n):
        return b"".join(self.read(addr + i, min(CHUNK, n - i)) for i in range(0, n, CHUNK))
