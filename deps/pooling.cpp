template <typename T>
void maxpooling(T *x, T *y, int *indexes, int xsize1, int xsize2, int xsize3,
    int winsize1, int winsize2, int pad1, int pad2, int stride1, int stride2) {

    int o = 0;
    for (int m3 = 0; m3 < xsize3; m3++) {
        for (int m2 = -pad2; m2 <= xsize2+pad2-winsize2; m2 += stride2) {
            for (int m1 = -pad1; m1 <= xsize1+pad1-winsize1; m1 += stride1) {
                int i0 = m1 + m2 * xsize1;
                T maxval = x[i0];
                int maxi = i0;
                for (int n2 = m2; n2 < m2 + winsize2; n2++) {
                    for (int n1 = m1; n1 < m1 + winsize1; n1++) {
                        int i = n1 + n2 * xsize1;
                        T val;
                        if (n1 >= 0 && n1 < xsize1 && n2 >= 0 && n2 < xsize2) val = x[i];
                        else val = 0;
                        if (val > maxval) {
                            maxval = val;
                            maxi = i;
                        }
                    }
                }
                indexes[o] = maxi;
                y[o] = maxval;
                o++;
            }
        }
    }
}

template <typename T>
void maxpooling_grad(T *gx, T *gy, int *indexes, int ysize) {
    for (int i = 0; i < ysize; i++) gx[indexes[i]] += gy[i];
}

#define POOLING_CAPI(NAME, T) \
void NAME(T *x, T *y, int *indexes, int xsize1, int xsize2, int xsize3, \
    int winsize1, int winsize2, int pad1, int pad2, int stride1, int stride2) { \
    maxpooling(x, y, indexes, xsize1, xsize2, xsize3, winsize1, winsize2, pad1, pad2, stride1, stride2); \
} \
void NAME ## _grad(T *gx, T *gy, int *indexes, int ysize) { \
    maxpooling_grad(gx, gy, indexes, ysize); \
}

extern "C" {
    POOLING_CAPI(maxpooling_f32, float)
}
