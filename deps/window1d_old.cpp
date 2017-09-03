#include <iostream>

template <typename T>
void window1d(const T *x, T *y, const int *xsize, const int batchsize,
    const int winsize, const int pad, const int stride, const int dilation) {

    int yi = 0;
    int s = 0;
    for (int b = 0; b < batchsize; b++) {
        int e = s + xsize[b];
        int i = s - pad;
        while (i+winsize <= e+pad) {
            for (int j = 0; j < winsize; j++) {
                int xi = i + j * dilation;
                if (xi >= s && xi < e) y[yi] = x[xi];
                else y[yi] = 0;
                yi++;
            }
            i += stride;
        }
        s = e;
    }
}

template <typename T>
void window1d_grad(const T *gy, T *gx, const int *xsize, const int batchsize,
    const int winsize, const int pad, const int stride, const int dilation) {

    int yi = 0;
    int s = 0;
    for (int b = 0; b < batchsize; b++) {
        int e = s + xsize[b];
        int i = s - pad;
        while (i+winsize <= e+pad) {
            for (int j = 0; j < winsize; j++) {
                int xi = i + j * dilation;
                if (xi >= s && xi < e) gx[xi] += gy[yi];
                yi++;
            }
            i += stride;
        }
        s = e;
    }
}

#define WINDOW1D_CAPI(T) \
void window1d ## _ ## T(T *x, T *y, int *xsize, int batchsize, int winsize, int pad, int stride, int dilation) { \
    window1d(x, y, xsize, batchsize, winsize, pad, stride, dilation); \
} \
void window1d_grad ## _ ## T(T *gy, T *gx, int *xsize, int batchsize, int winsize, int pad, int stride, int dilation) { \
    window1d_grad(gy, gx, xsize, batchsize, winsize, pad, stride, dilation); \
} \

extern "C" {
    WINDOW1D_CAPI(float)
}
