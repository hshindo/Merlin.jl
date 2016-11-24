#include <algorithm>
#include <stdio.h>
#include <stdint.h>

template <typename T>
void im2col(T *x, T *y, int xsize1, int xsize2, int xsize3, int winsize1, int winsize2,
    int pad1, int pad2, int stride1, int stride2) {

    int n1 = (xsize1 + 2 * pad1 - winsize1) / stride1 + 1;
    int n2 = (xsize2 + 2 * pad2 - winsize2) / stride2 + 1;
    int o = 0;
    for (int d3 = 0; d3 < xsize3; d3++) {
        for (int d2 = 0; d2 < winsize2; d2++) {
            for (int d1 = 0; d1 < winsize1; d1++) {
                for (int k2 = 0; k2 < n2; k2++) {
                    for (int k1 = 0; k1 < n1; k1++) {
                        int i1 = k1 * stride1 - pad1 + d1;
                        int i2 = k2 * stride2 - pad2 + d2;
                        if (i1 >= 0 && i1 < xsize1 && i2 >= 0 && i2 < xsize2) {
                            int i = i1 + xsize1 * (i2 + xsize2*d3);
                            y[o] = x[i];
                        }
                        else y[o] = 0;
                        o++;
                    }
                }
            }
        }
    }
}

template <typename T>
void im2col_grad(T *gy, T *gx, int xsize1, int xsize2, int xsize3, int winsize1, int winsize2,
    int pad1, int pad2, int stride1, int stride2) {

    int n1 = (xsize1 + 2 * pad1 - winsize1) / stride1 + 1;
    int n2 = (xsize2 + 2 * pad2 - winsize2) / stride2 + 1;
    int o = 0;
    for (int d3 = 0; d3 < xsize3; d3++) {
        for (int d2 = 0; d2 < winsize2; d2++) {
            for (int d1 = 0; d1 < winsize1; d1++) {
                for (int k2 = 0; k2 < n2; k2++) {
                    for (int k1 = 0; k1 < n1; k1++) {
                        int i1 = k1*stride1 - pad1 + d1;
                        int i2 = k2*stride2 - pad2 + d2;
                        if (i1 >= 0 && i1 < xsize1 && i2 >= 0 && i2 < xsize2) {
                            int i = i1 + xsize1 * (i2 + xsize2*d3);
                            gx[i] += gy[o];
                        }
                        o++;
                    }
                }
            }
        }
    }
}

#define IM2COL_CAPI(NAME, T) \
void NAME(T *x, T *y, int xsize1, int xsize2, int xsize3, int winsize1, int winsize2, \
    int pad1, int pad2, int stride1, int stride2) { \
    im2col(x, y, xsize1, xsize2, xsize3, winsize1, winsize2, pad1, pad2, stride1, stride2); \
} \
void NAME ## _ ## grad(T *gy, T *gx, int xsize1, int xsize2, int xsize3, int winsize1, int winsize2, \
    int pad1, int pad2, int stride1, int stride2) { \
    im2col_grad(gy, gx, xsize1, xsize2, xsize3, winsize1, winsize2, pad1, pad2, stride1, stride2); \
}

extern "C" {
    IM2COL_CAPI(im2col_f32, float)
}
