template <typename T>
void window1d(const T *x, T *y, const int height,
    const int dim_h, const int pad_h, const int stride_h) {
    const int size_h = (height + 2*pad_h - dim_h) / stride_h + 1;

    //#pragma omp parallel for
    for (int n = 0; n < size_h; n++) {
        for (int h = 0; h < dim_h; h++) {
            int yh = n * dim_h + h;
            int xh = -pad_h + stride_h * n + h;
            y[yh] = (xh >= 0 && xh < height) ? x[xh] : 0;
        }
    }
}

template <typename T>
void window1d_grad(const T *gy, T *gx, const int height,
    const int dim_h, const int pad_h, const int stride_h) {
    const int size_h = (height + 2*pad_h - dim_h) / stride_h + 1;

    //#pragma omp parallel for
    for (int n = 0; n < size_h; n++) {
        for (int h = 0; h < dim_h; h++) {
            int yh = n * dim_h + h;
            int xh = -pad_h + stride_h * n + h;
            if (xh >= 0 && xh < height) gx[xh] += gy[yh];
        }
    }
}

#define WINDOW1D_CAPI(NAME, T) \
void NAME(T *x, T *y, int height, int dim_h, int pad_h, int stride_h) { \
    window1d(x, y, height, dim_h, pad_h, stride_h); \
} \
void NAME ## _ ## grad(T *gy, T *gx, int height, int dim_h, int pad_h, int stride_h) { \
    window1d_grad(gy, gx, height, dim_h, pad_h, stride_h); \
} \

extern "C" {
    WINDOW1D_CAPI(window1d_f32, float)
}
