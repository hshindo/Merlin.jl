#include <algorithm>
#include <stdio.h>
#include <stdint.h>

template <typename T>
void window1d(T *x, T *y, int l, int w, int s, int p) {
    int yi = 0;
    for (int i = -p; i <= l+p-w; i+= s) {
        for (int j = i; j < i+w; j++) {
            int xi = j;
            y[yi] = (j >= 0 && j < l) ? x[xi] : 0;
            yi++;
        }
    }
}

template <typename T>
void window2d(T *x, T *y, int l1, int l2, int w1, int w2, int s1, int s2, int p1, int p2) {
    int yi = 0;
    for (int i2 = -p2; i2 <= l2+p2-w2; i2 += s2) {
        for (int i1 = -p1; i1 <= l1+p1-w1; i1 += s1) {
            for (int j2 = i2; j2 < i2+w2; j2++) {
                for (int j1 = i1; j1 < i1+w1; j1++) {
                    int xi = j1 + j2 * l1;
                    if (j1 >= 0 && j1 < l1 && j2 >= 0 && j2 < l2) y[yi] = x[xi];
                    else y[yi] = 0.0;
                    yi++;
                }
            }
        }
    }
}

template <typename T>
void window1d_grad(T *gx, T *gy, int l, int w, int s, int p) {
    for (int n = 0; n <= (l+2*p-w)/s; n++) {
        for (int i = 0; i < w; i++) {
            int k = -p + s * n + i;
            if (k >= 0 && k < l) gx[k] += gy[w*n+i];
        }
    }
}

template <typename T>
void window2d_grad(T *gx, T *gy, int l1, int l2, int w1, int w2, int s1, int s2, int p1, int p2) {
    int yi = 0;
    for (int i2 = -p2; i2 <= l2+p2-w2; i2 += s2) {
        for (int i1 = -p1; i1 <= l1+p1-w1; i1 += s1) {
            for (int j2 = i2; j2 < i2+w2; j2++) {
                for (int j1 = i1; j1 < i1+w1; j1++) {
                    int xi = j1 + j2 * l1;
                    if (j1 >= 0 && j1 < l1 && j2 >= 0 && j2 < l2) gx[xi] += gy[yi];
                    yi++;
                }
            }
        }
    }
}

#define WINDOW1D_CAPI(NAME, T) \
void NAME(T *x, T *y, int l, int w, int s, int p) { \
    window1d(x, y, l, w, s, p); \
} \
void NAME ## _ ## grad(T *gx, T *gy, int l, int w, int s, int p) { \
    window1d_grad(gx, gy, l, w, s, p); \
}

#define WINDOW2D_CAPI(NAME, T) \
void NAME(T *x, T *y, int l1, int l2, int w1, int w2, int s1, int s2, int p1, int p2) { \
    window2d(x, y, l1, l2, w1, w2, s1, w2, p1, p2); \
} \
void NAME ## _ ## grad(T *gx, T *gy, int l1, int l2, int w1, int w2, int s1, int s2, int p1, int p2) { \
    window2d_grad(gx, gy, l1, l2, w1, w2, s1, w2, p1, p2); \
}

extern "C" {
    WINDOW1D_CAPI(window1d_f32, float)
    WINDOW1D_CAPI(window1d_f64, double)
    WINDOW1D_CAPI(window1d_i64, int64_t)
    WINDOW2D_CAPI(window2d_f32, float)
    WINDOW2D_CAPI(window2d_f64, double)
    WINDOW2D_CAPI(window2d_i64, int64_t)
}
