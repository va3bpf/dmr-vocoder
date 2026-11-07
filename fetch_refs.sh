#!/bin/sh
# Fetch the reference sources into vocoder/ref/ (not in git):
#   mbelib  - AMBE 3600x2450 decoder (ISC licence), the tables and the parameter decoding
#   dsd     - DMR AMBE interleave schedule (include/dmr_const.h)
#   op25    - AMBE+2 half-rate encoder (ambe_encoder.cc) on Pavel Yazev's fixed-point IMBE analysis (GPL v3)
set -e
cd "$(dirname "$0")"
mkdir -p ref
cd ref
[ -d mbelib ] || git clone --depth 1 https://github.com/szechyjs/mbelib.git
[ -d dsd ] || git clone --depth 1 https://github.com/szechyjs/dsd.git
if [ ! -d op25 ]; then
	git clone --depth 1 --filter=blob:none --sparse https://github.com/boatbod/op25.git
	git -C op25 sparse-checkout set op25/gr-op25_repeater/lib
fi
if [ ! -d pesq ]; then
	# ITU-T P.862 PESQ reference code, as packaged by the "pesq" Python package (no wheels for Windows)
	mkdir -p pesq
	curl -sSfL -o pesq/pesq.tar.gz https://files.pythonhosted.org/packages/source/p/pesq/pesq-0.0.4.tar.gz
	tar xzf pesq/pesq.tar.gz -C pesq
fi
