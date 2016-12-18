#include <limits>

template <typename T>
void maxpooling1d(T *x, T *y, int *indexes, int xsize1, int xsize2, int winsize, int pad, int stride) {
    int o = 0;
    for (int m2 = 0; m2 < xsize2; m2++) {
        for (int m1 = -pad; m1 <= xsize1+pad-winsize; m1 += stride) {
            T maxval = std::numeric_limits<T>::min();
            int maxi = -1;

            for (int n1 = m1; n1 < m1 + winsize; n1++) {
                int i = n1 + m2 * xsize1;
                T val = (n1 >= 0 && n1 < xsize1) ? x[i] : 0;
                if (val > maxval) {
                    maxval = val;
                    maxi = i;
                }
            }
            indexes[o] = maxi;
            y[o] = maxval;
            o++;
        }
    }
}

template <typename T>
void maxpooling2d(T *x, T *y, int *indexes, int xsize1, int xsize2, int xsize3,
    int winsize1, int winsize2, int pad1, int pad2, int stride1, int stride2) {
    int o = 0;
    for (int m3 = 0; m3 < xsize3; m3++) {
        for (int m2 = -pad2; m2 <= xsize2+pad2-winsize2; m2 += stride2) {
            for (int m1 = -pad1; m1 <= xsize1+pad1-winsize1; m1 += stride1) {
                T maxval = std::numeric_limits<T>::min();
                int maxi = -1;

                for (int n2 = m2; n2 < m2 + winsize2; n2++) {
                    for (int n1 = m1; n1 < m1 + winsize1; n1++) {
                        int i = n1 + n2*xsize1 + m3*xsize1*xsize2;
                        T val = (n1 >= 0 && n1 < xsize1 && n2 >= 0 && n2 < xsize2) ? x[i] : 0;
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
void maxpooling_grad(T *gy, T *gx, int *indexes, int ysize) {
    for (int i = 0; i < ysize; i++) gx[indexes[i]] += gy[i];
}

extern "C" {
    void maxpooling1d_f32(float* x, float *y, int *indexes, int xsize1, int xsize2,
        int winsize, int pad, int stride) {
        maxpooling1d(x, y, indexes, xsize1, xsize2, winsize, pad, stride);
    }
    void maxpooling2d_f32(float* x, float *y, int *indexes, int xsize1, int xsize2, int xsize3,
        int winsize1, int winsize2, int pad1, int pad2, int stride1, int stride2) {
        maxpooling2d(x, y, indexes, xsize1, xsize2, xsize3, winsize1, winsize2, pad1, pad2, stride1, stride2);
    }
    void maxpooling_f32_grad(float *gy, float *gx, int *indexes, int ysize) {
        maxpooling_grad(gy, gx, indexes, ysize);
    }
}
