#!/bin/bash
# Run md380-emu (tools/build_md380emu.sh) on files from Windows paths under /mnt:
#   wsl -u root -- bash /mnt/e/exam/OpenDMR/vocoder/tools/md380emu.sh enc|dec IN OUT
# enc: 8 kHz 16-bit mono raw PCM -> .amb (DSD: ".amb", then 8 bytes a frame: status, 49 bits MSB first, the last
# bit as the LSB of byte 8). dec: .amb -> raw PCM.
set -e
EMU=/root/md380tools/emulator/md380-emu
case "$1" in
	enc) exec qemu-arm-static "$EMU" -e -i "$2" -o "$3" ;;
	dec) exec qemu-arm-static "$EMU" -d -i "$2" -o "$3" ;;
	*) echo "usage: $0 enc|dec IN OUT" >&2; exit 2 ;;
esac
