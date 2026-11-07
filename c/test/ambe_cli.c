/*
 * The C vocoder on the PC, for the tests (py/c_eval.py):
 *   ambe_cli enc in.raw out.bit     16-bit 8 kHz PCM -> 9 bytes a frame
 *   ambe_cli dec in.bit out.raw
 */
#include <stdio.h>
#include <string.h>

#include "ambe.h"

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: ambe_cli enc|dec in out\n");
        return 2;
    }
    FILE *fi = fopen(argv[2], "rb"), *fo = fopen(argv[3], "wb");
    if (!fi || !fo) {
        fprintf(stderr, "cannot open files\n");
        return 1;
    }
    if (!strcmp(argv[1], "enc")) {
        static ambe_encoder enc;
        ambe_enc_init(&enc);
        int16_t pcm[AMBE_N];
        uint8_t out[9];
        while (fread(pcm, 2, AMBE_N, fi) == AMBE_N) {
            if (ambe_enc_frame(&enc, pcm, out)) {
                fwrite(out, 1, 9, fo);
            }
        }
        memset(pcm, 0, sizeof(pcm));
        for (int i = 0; i < AMBE_ENC_DELAY; i++) {
            if (ambe_enc_frame(&enc, pcm, out)) {
                fwrite(out, 1, 9, fo);
            }
        }
    } else {
        static ambe_decoder dec;
        ambe_dec_init(&dec);
        uint8_t in[9];
        int16_t pcm[AMBE_N];
        while (fread(in, 1, 9, fi) == 9) {
            ambe_dec_frame(&dec, in, pcm);
            fwrite(pcm, 2, AMBE_N, fo);
        }
    }
    fclose(fi);
    fclose(fo);
    return 0;
}
