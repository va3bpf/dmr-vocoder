#!/bin/bash
# Build md380tools' md380-emu in WSL (Ubuntu 22.04; run as root): the Tytera MD-380 firmware D002.032's DVSI AMBE+2
# vocoder under qemu-user, a reference encoder / decoder.
# It is an external test tool: nothing of it goes into this repository or the firmware.
#
#   wsl -u root -- bash /mnt/e/exam/OpenDMR/vocoder/tools/build_md380emu.sh
#
# Needs: apt install gcc-arm-linux-gnueabi libc6-dev-armel-cross binutils-arm-linux-gnueabi qemu-user-static git make
# Result: /root/md380tools/emulator/md380-emu (run with qemu-arm-static); py/md380emu.py drives it.
set -e
cd /root
[ -d md380tools ] || git clone -q --depth 1 https://github.com/travisgoodspeed/md380tools.git
cd md380tools/firmware
# downloads the original firmware zip, checks it and unwraps it
[ -s unwrapped/D002.032.img ] || make -s -f Makefile_orig unwrapped/D002.032.img
cd ../emulator
# 1) the firmware is unwrapped already: no recursive make of every firmware
sed -i 's/^\t"\${MAKE}" -C ..\/firmware$/\t@true/' Makefile
# 2) the .amb magic check: terminate the 4 bytes read (packed[4], not [5]) and fail when they differ, not match
sed -i 's/packed\[5\]=0;/packed[4]=0;/; s/if(!strcmp(packed,".amb")){/if(strcmp((char*)packed,".amb")){/' ambe.c
# 3) the encoder drops its first 26 frames ("ignore start noise"): keep them, so frame k is input frame k
sed -i 's/if (frame > 25) \/\/ skip the first ambe frames/if (frame >= 0) \/\/ (was > 25) skip the first ambe frames/' ambe.c
grep -q 'packed\[4\]=0;' ambe.c
grep -q 'if (frame >= 0)' ambe.c
make -s md380-emu
ls -la md380-emu
echo "built /root/md380tools/emulator/md380-emu"
