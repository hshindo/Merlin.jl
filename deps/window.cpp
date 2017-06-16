#include <iostream>

template <typename T>
void window1d(const T *x, const int *xsize, const int batchsize, T *y,
    const int ksize, const int pad, const int stride) {

    //int *offset = new int[batchsize];
    //offset[0] = 0;
    //for (int i = 1; i < batchsize; i++) offset[i] = offset[i-1] + xsize[i-1];

    int offset = 0;
    int yh = 0;
    for (int i = 0; i < batchsize; i++) {
        int count = (xsize[i] + 2 * pad - ksize) / stride + 1;
        for (int n = 0; n < count; n++) {
            for (int h = 0; h < ksize; h++) {
                //int yh = (i * count * ksize) + (n * ksize) + h;
                int xh = offset - pad + n * stride + h;
                y[yh] = (xh >= offset && xh < offset+xsize[i]) ? x[xh] : 0;
                yh += 1;
            }
        }
        offset += xsize[i];
    }
    //delete[] offset;
}

template <typename T>
void window1d_grad(const T *gy, T *gx, const int *gxsize, const int batchsize,
    const int ksize, const int pad, const int stride) {

    int offset = 0;
    int yh = 0;
    for (int i = 0; i < batchsize; i++) {
        int count = (gxsize[i] + 2 * pad - ksize) / stride + 1;
        for (int n = 0; n < count; n++) {
            for (int h = 0; h < ksize; h++) {
                //int yh = (i * count * ksize) + (n * ksize) + h;
                int xh = offset - pad + n * stride + h;
                if (xh >= offset && xh < offset+gxsize[i]) gx[xh] += gy[yh];
                yh += 1;
            }
        }
        offset += gxsize[i];
    }
}

#define WINDOW1D_CAPI(T) \
void window1d ## _ ## T(T *x, int *xsize, int batchsize, T *y, int ksize, int pad, int stride) { \
    window1d(x, xsize, batchsize, y, ksize, pad, stride); \
} \
void window1d_grad ## _ ## T(T *gy, T *gx, int *xsize, int batchsize, int ksize, int pad, int stride) { \
    window1d_grad(gy, gx, xsize, batchsize, ksize, pad, stride); \
} \

extern "C" {
    WINDOW1D_CAPI(float)
}
