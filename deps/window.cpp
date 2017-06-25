template <typename T>
void window1d(const T *x, T *y, const int xsize,
    const int winsize, const int pad, const int stride) {

    const int count = (xsize + 2*pad - winsize) / stride + 1;
    for (int n = 0; n < count; n++) {
        for (int i = 0; i < winsize; i++) {
            int yi = n * winsize + i;
            int xi = -pad + stride * n + i;
            y[yi] = (xi >= 0 && xi < xsize) ? x[xi] : 0;
        }
    }
}

template <typename T>
void window1d_grad(const T *gy, T *gx, const int xsize,
    const int winsize, const int pad, const int stride) {

    const int count = (xsize + 2*pad - winsize) / stride + 1;
    for (int n = 0; n < count; n++) {
        for (int i = 0; i < winsize; i++) {
            int yi = n * winsize + i;
            int xi = -pad + stride * n + i;
            if (xi >= 0 && xi < xsize) gx[xi] += gy[yi];
        }
    }
}

#define WINDOW1D_CAPI(T) \
void window1d ## _ ## T(T *x, T *y, int xsize, int winsize, int pad, int stride) { \
    window1d(x, y, xsize, winsize, pad, stride); \
} \
void window1d_grad ## _ ## T(T *gy, T *gx, int xsize, int winsize, int pad, int stride) { \
    window1d_grad(gy, gx, xsize, winsize, pad, stride); \
} \

extern "C" {
    WINDOW1D_CAPI(float)
}
