/*
 * The encoder's harmonic fit in C (py/ambe_analysis.py Analyzer.fit, which calls this through ctypes when
 * bin/vocoder_c.dll is there; c/build.sh builds it). Doubles throughout, the same arithmetic as the Python.
 *
 * For each harmonic l = 1..L of w0 (rad / sample), over the spectrum bins k with (l - 0.5) w0 <= w_k < (l + 0.5) w0
 * (w_k = 2 pi k / nfft, 0 < w_k < pi not required): the window's transform W shifted to l w0, the least-squares
 * amplitude A_l = sum conj(W) S / sum |W|^2, the band's energy sum |S|^2 and the error sum |S - A_l W|^2.
 * W at any frequency is linearly interpolated in a table of the window's transform (nwt points over 2 pi).
 */
#include <math.h>

#define EXPORT __declspec(dllexport)

static void window_at(double dw, const double *wt_re, const double *wt_im, int nwt, double *re, double *im)
{
    double two_pi = 2.0 * M_PI;
    double m = fmod(dw, two_pi);
    if (m < 0) {
        m += two_pi;
    }
    double pos = m / two_pi * nwt;
    long i = (long)floor(pos);
    double f = pos - i;
    long i0 = i % nwt, i1 = (i + 1) % nwt;
    *re = wt_re[i0] * (1 - f) + wt_re[i1] * f;
    *im = wt_im[i0] * (1 - f) + wt_im[i1] * f;
}

// A_re, A_im, err, en: L + 1 entries each, index 0 unused (left 0)
EXPORT void harmonic_fit(const double *s_re, const double *s_im, int nbins, int nfft, const double *wt_re,
                         const double *wt_im, int nwt, double w0, int L, double *A_re, double *A_im, double *err,
                         double *en)
{
    for (int l = 0; l <= L; l++) {
        A_re[l] = A_im[l] = err[l] = en[l] = 0;
    }
    for (int l = 1; l <= L; l++) {
        double lo = (l - 0.5) * w0, hi = (l + 0.5) * w0;
        int k0 = (int)floor(lo * nfft / (2.0 * M_PI)) - 1;
        if (k0 < 0) {
            k0 = 0;
        }
        double nr = 0, ni = 0, ww = 0, e = 0;
        int first = -1, last = -1;
        for (int k = k0; k < nbins; k++) {
            double wk = ((double)k * 2.0) * M_PI / nfft;
            if (wk >= hi) {
                break;
            }
            if (wk < lo) {
                continue;
            }
            if (first < 0) {
                first = k;
            }
            last = k;
            double wr, wi;
            window_at(wk - l * w0, wt_re, wt_im, nwt, &wr, &wi);
            nr += wr * s_re[k] + wi * s_im[k]; // conj(W) S
            ni += wr * s_im[k] - wi * s_re[k];
            ww += wr * wr + wi * wi;
            e += s_re[k] * s_re[k] + s_im[k] * s_im[k];
        }
        if (first < 0 || ww <= 0) {
            continue;
        }
        double ar = nr / ww, ai = ni / ww;
        double er = 0;
        for (int k = first; k <= last; k++) {
            double wk = ((double)k * 2.0) * M_PI / nfft;
            double wr, wi;
            window_at(wk - l * w0, wt_re, wt_im, nwt, &wr, &wi);
            double dr = s_re[k] - (ar * wr - ai * wi);
            double di = s_im[k] - (ar * wi + ai * wr);
            er += dr * dr + di * di;
        }
        A_re[l] = ar;
        A_im[l] = ai;
        en[l] = e;
        err[l] = er;
    }
}
