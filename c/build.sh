#!/bin/sh
# Build the host tools in vocoder/bin with the GCC in toolchain/winlibs (WinLibs MinGW-w64, unpacked there; not in git):
#   pesq_nb.exe   narrow-band PESQ of two raw files (c/pesq_main.c + the ITU P.862 code of the "pesq" package,
#                 fetched into ref/pesq by fetch_refs.sh; research and evaluation use only)
set -e
cd "$(dirname "$0")/.."
export PATH="$(pwd)/../toolchain/winlibs/mingw64/bin:$PATH"
mkdir -p bin
P=ref/pesq/pesq-0.0.4/pesq
gcc -O2 -w -static -I$P -o bin/pesq_nb.exe c/pesq_main.c $P/dsp.c $P/pesqdsp.c $P/pesqmod.c -lm
echo "built bin/pesq_nb.exe"
#   vocoder_c.dll speed-ups for the Python (c/analysis.c: the encoder's harmonic fit), loaded with ctypes
gcc -O2 -shared -static-libgcc -o bin/vocoder_c.dll c/analysis.c -lm
echo "built bin/vocoder_c.dll"
#   ambe_cli.exe  our C vocoder (c/ambe) on the PC: encode / decode files (c/test/ambe_cli.c)
gcc -O2 -Wall -Wextra -static -o bin/ambe_cli.exe -Ic/ambe c/test/ambe_cli.c c/ambe/ambe_frame.c c/ambe/ambe_parms.c \
    c/ambe/ambe_dec.c c/ambe/ambe_enc.c c/ambe/ambe_tables.c -lm
echo "built bin/ambe_cli.exe"
#   ambe_ref.exe  the double reference (c/ref, bit-exact with the Python), the same command line
gcc -O2 -Wall -Wextra -static -o bin/ambe_ref.exe -Ic/ref c/test/ambe_cli.c c/ref/ambe_frame.c c/ref/ambe_parms.c \
    c/ref/ambe_dec.c c/ref/ambe_enc.c c/ref/ambe_tables.c -lm
echo "built bin/ambe_ref.exe"
