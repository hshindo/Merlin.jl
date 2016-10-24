#include <algorithm>
#include <stdio.h>
#include <stdint.h>

template <typename T>
void window1d(T *x, T *y, int xsize, int winsize, int pad, int stride) {
    int o = 0;
    for (int d = -pad; d <= xsize+pad-winsize; d += stride) {
        for (int i = d; i < d+winsize; i++) {
            y[o] = (i >= 0 && i < xsize) ? x[i] : 0;
            o++;
        }
    }
}

template <typename T>
void window1d_grad(T *gx, T *gy, int xsize, int winsize, int pad, int stride) {
    int o = 0;
    for (int d = -pad; d <= xsize+pad-winsize; d += stride) {
        for (int i = d; i < d+winsize; i++) {
            if (i >= 0 && i < xsize) gx[i] += gy[o];
            o++;
        }
    }
}

#define WINDOW1D_CAPI(NAME, T) \
void NAME(T *x, T *y, int xsize, int winsize, int pad, int stride) { \
    window1d(x, y, xsize, winsize, pad, stride); \
} \
void NAME ## _ ## grad(T *gx, T *gy, int xsize, int winsize, int pad, int stride) { \
    window1d_grad(gx, gy, xsize, winsize, pad, stride); \
}

extern "C" {
    WINDOW1D_CAPI(window1d_f32, float)
    WINDOW1D_CAPI(window1d_i64, int64_t)
}
