template <typename T>
void maxpooling2d(const T *x, T *y, int *inds,
    const int height, const int width, const int num,
    const int dim_h, const int dim_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w) {
    const int size_h = (height + 2*pad_h - dim_h) / stride_h + 1;
    const int size_w = (width + 2*pad_w - dim_w) / stride_w + 1;

    //#pragma omp parallel for
    for (int n = 0; n < num; n++) {
        for (int yw = 0; yw < size_w; yw++) {
            for (int yh = 0; yh < size_h; yh++) {
                int yi = yh + yw * size_h + n * size_h * size_w;
                int maxi = -1;
                for (int w = 0; w < dim_w; w++) {
                    int xw = -pad_w + stride_w * yw + w;
                    if (xw < 0 || xw >= width) continue;
                    for (int h = 0; h < dim_h; h++) {
                        int xh = -pad_h + stride_h * yh + h;
                        if (xh < 0 || xh > height) continue;
                        int xi = xh + xw * height + n * height * width;
                        if (maxi < 0 || x[xi] > x[maxi]) maxi = xi;
                    }
                }
                y[yi] = x[maxi];
                inds[yi] = maxi;
            }
        }
    }
}

template <typename T>
void maxpooling_grad(const T *gy, T *gx, int *inds, int size_y) {
    for (int i = 0; i < size_y; i++) gx[inds[i]] += gy[i];
}

template <typename T>
void avgpooling2d(const T *x, T *y,
    const int height, const int width, const int num,
    const int dim_h, const int dim_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w) {
    const int size_h = (height + 2*pad_h - dim_h) / stride_h + 1;
    const int size_w = (width + 2*pad_w - dim_w) / stride_w + 1;

    //#pragma omp parallel for
    for (int n = 0; n < num; n++) {
        for (int yw = 0; yw < size_w; yw++) {
            for (int yh = 0; yh < size_h; yh++) {
                int yi = yh + yw * size_h + n * size_h * size_w;
                T sum = 0;
                for (int w = 0; w < dim_w; w++) {
                    int xw = -pad_w + stride_w * yw + w;
                    if (xw < 0 || xw >= width) continue;
                    for (int h = 0; h < dim_h; h++) {
                        int xh = -pad_h + stride_h * yh + h;
                        if (xh < 0 || xh > height) continue;
                        int xi = xh + xw * height + n * height * width;
                        sum += x[xi];
                    }
                }
                y[yi] = sum / (dim_h * dim_w);
            }
        }
    }
}

template <typename T>
void avgpooling2d_grad(const T *gy, T *gx,
    const int height, const int width, const int num,
    const int dim_h, const int dim_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w) {
    const int size_h = (height + 2*pad_h - dim_h) / stride_h + 1;
    const int size_w = (width + 2*pad_w - dim_w) / stride_w + 1;

    //#pragma omp parallel for
    for (int n = 0; n < num; n++) {
        for (int yw = 0; yw < size_w; yw++) {
            for (int yh = 0; yh < size_h; yh++) {
                int yi = yh + yw * size_h + n * size_h * size_w;
                for (int w = 0; w < dim_w; w++) {
                    int xw = -pad_w + stride_w * yw + w;
                    if (xw < 0 || xw >= width) continue;
                    for (int h = 0; h < dim_h; h++) {
                        int xh = -pad_h + stride_h * yh + h;
                        if (xh < 0 || xh > height) continue;
                        int xi = xh + xw * height + n * height * width;
                        gx[xi] += gy[yi] / (dim_h * dim_w);
                    }
                }
            }
        }
    }
}

#define MAXPOOLING2D_CAPI(NAME, T) \
void NAME(const T *x, T *y, int *inds, \
    int height, int width, int num, int dim_h, int dim_w, \
    int pad_h, int pad_w, int stride_h, int stride_w) { \
    maxpooling2d(x, y, inds, height, width, num, \
        dim_h, dim_w, pad_h, pad_w, stride_h, stride_w); \
} \
void NAME ## _ ## grad(T *gy, T *gx, int *inds, int size_y) { \
    maxpooling_grad(gy, gx, inds, size_y); \
}

#define AVGPOOLING2D_CAPI(NAME, T) \
void NAME(const T *x, T *y, \
    int height, int width, int num, int dim_h, int dim_w, \
    int pad_h, int pad_w, int stride_h, int stride_w) { \
    avgpooling2d(x, y, height, width, num, \
        dim_h, dim_w, pad_h, pad_w, stride_h, stride_w); \
} \
void NAME ## _ ## grad(T *gy, T *gx, \
    int height, int width, int num, int dim_h, int dim_w, \
    int pad_h, int pad_w, int stride_h, int stride_w) { \
    avgpooling2d_grad(gy, gx, height, width, num, \
        dim_h, dim_w, pad_h, pad_w, stride_h, stride_w); \
} \

extern "C" {
    MAXPOOLING2D_CAPI(maxpooling2d_f32, float)
    AVGPOOLING2D_CAPI(avgpooling2d_f32, float)
}
