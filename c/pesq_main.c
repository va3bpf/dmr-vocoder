/*
 * pesq_nb ref.raw deg.raw [rate]: narrow-band PESQ (ITU-T P.862, MOS-LQO as P.862.1) of two 16-bit little-endian
 * mono raw files, printed on stdout. A wrapper around the ITU reference code as packaged by the "pesq" Python
 * package (MIT wrapper, ITU licence for the algorithm: research and evaluation only). The code is not in this
 * repository: vocoder/c/build.sh fetches it into vocoder/ref/pesq.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pesq.h"
#include "pesqio.h"
#include "pesqmain.h"

static float *load(const char *path, long *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    *n = bytes / 2;
    short *s = malloc(*n * sizeof(short));
    float *x = malloc(*n * sizeof(float));
    if (fread(s, sizeof(short), *n, f) != (size_t)*n) {
        *n = 0;
    }
    fclose(f);
    for (long i = 0; i < *n; i++) {
        x[i] = s[i];
    }
    free(s);
    return x;
}

static void setup(SIGNAL_INFO *info, const char *name, float *data, long n)
{
    memset(info, 0, sizeof(*info));
    strncpy(info->path_name, name, sizeof(info->path_name) - 1);
    strncpy(info->file_name, name, sizeof(info->file_name) - 1);
    info->Nsamples = n;
    info->apply_swap = 0;
    info->input_filter = 1;
    info->data = data;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: pesq_nb ref.raw deg.raw [rate]\n");
        return 2;
    }
    long rate = (argc > 3) ? atol(argv[3]) : 8000;
    long err = 0;
    char *err_type = "unknown";
    select_rate(rate, &err, &err_type);
    if (err) {
        fprintf(stderr, "rate: %s\n", err_type);
        return 1;
    }
    long nr, nd;
    float *ref = load(argv[1], &nr);
    float *deg = load(argv[2], &nd);
    if (!ref || !deg || !nr || !nd) {
        fprintf(stderr, "cannot read the input\n");
        return 1;
    }
    SIGNAL_INFO ri, di;
    ERROR_INFO ei;
    setup(&ri, "ref", ref, nr);
    setup(&di, "deg", deg, nd);
    memset(&ei, 0, sizeof(ei));
    ei.mode = NB_MODE;
    pesq_measure(&ri, &di, &ei, &err, &err_type);
    if (err) {
        fprintf(stderr, "pesq: %ld %s\n", err, err_type);
        return 1;
    }
    printf("%.4f %.4f\n", ei.mapped_mos, ei.pesq_mos);
    return 0;
}
