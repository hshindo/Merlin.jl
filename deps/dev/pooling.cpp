#include "carray.h"

template <typename T>
void maxpooling2d(T *x, int *size_x, T *y, int *maxidxs, int *winsize, int *stride, int *padsize) {
    int x1 = size_x[0], x2 = size_x[1], x3 = size_x[2];
    //int x1 = x.dims[0], x2 = x.dims[1];
    int w1 = winsize[0], w2 = winsize[1];
    int s1 = stride[0], s2 = stride[1];
    int p1 = padsize[0], p2 = padsize[1];

    int o = 0;
    for (int m3 = 0; m3 < x3; m3++) {
        for (int m2 = -p2; m2 <= x2+p2-w2; m2 += s2) {
            for (int m1 = -p1; m1 <= x1+p1-w1; m1 += s1) {
                int i0 = m1 + m2 * x1;
                T maxval = x[i0];
                int maxi = i0;
                for (int n2 = m2; n2 < m2 + w2; n2++) {
                    for (int n1 = m1; n1 < m1 + w1; n1++) {
                        int i = n1 + n2 * x1;
                        T val;
                        if (n1 >= 0 && n1 < x1 && n2 >= 0 && n2 < x2) val = x[i];
                        else val = 0;
                        if (val > maxval) {
                            maxval = val;
                            maxi = i;
                        }
                    }
                }
                maxidxs[o] = maxi;
                y[o] = maxval;
                o++;
            }
        }
    }
}

template <typename T>
void maxpooling2d_grad(T *gx, T *gy, int *maxidxs, int size_y) {
    for (int i = 0; i < size_y; i++) gx[maxidxs[i]] += gy[i];
}

#define POOLING_CAPI(T) \
void maxpooling2d ## _ ## T(T *x, int *size_x, T *y, int *maxidxs, \
    int *winsize, int *stride, int *padsize) { \
    maxpooling2d(x, size_x, y, maxidxs, winsize, stride, padsize); \
} \
void maxpooling2d ## _ ## grad ## _ ## T(T *gx, T *gy, int *maxidxs, int size_y) { \
    maxpooling2d_grad(gx, gy, maxidxs, size_y); \
}

extern "C" {
    POOLING_CAPI(float)
    POOLING_CAPI(double)
}
