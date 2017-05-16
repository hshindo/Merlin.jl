template <typename T>
void window1d(const T *x, T *y, const int height,
    const int ksize_h, const int pad_h, const int stride_h) {
    const int size_h = (height + 2*pad_h - ksize_h) / stride_h + 1;

    for (int n = 0; n < size_h; n++) {
        for (int h = 0; h < ksize_h; h++) {
            int yh = n * ksize_h + h;
            int xh = -pad_h + stride_h * n + h;
            y[yh] = (xh >= 0 && xh < height) ? x[xh] : 0;
        }
    }
}

template <typename T>
void window1d_grad(const T *gy, T *gx, const int height,
    const int ksize_h, const int pad_h, const int stride_h) {
    const int size_h = (height + 2*pad_h - ksize_h) / stride_h + 1;

    for (int n = 0; n < size_h; n++) {
        for (int h = 0; h < ksize_h; h++) {
            int yh = n * ksize_h + h;
            int xh = -pad_h + stride_h * n + h;
            if (xh >= 0 && xh < height) gx[xh] += gy[yh];
        }
    }
}

#define WINDOW1D_CAPI(NAME, T) \
void NAME(T *x, T *y, int height, int ksize_h, int pad_h, int stride_h) { \
    window1d(x, y, height, ksize_h, pad_h, stride_h); \
} \
void NAME ## _ ## grad(T *gy, T *gx, int height, int ksize_h, int pad_h, int stride_h) { \
    window1d_grad(gy, gx, height, ksize_h, pad_h, stride_h); \
} \

extern "C" {
    WINDOW1D_CAPI(window1d_f32, float)
}
