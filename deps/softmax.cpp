#include <math.h>

inline int getindex(int i, int j, int k, int dim1, int dim2) {
    return i + dim1 * (k + dim2*j);
}

template<typename T>
void softmax(T *x, T *y, int dim1, int dim2, int dim3) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim3; j++) {
            T maxv = 0;
            for (int k = 0; k < dim2; k++) {
                int idx = getindex(i, j, k, dim1, dim2);
                if (k == 0 || x[idx] > maxv) maxv = x[idx];
            }

            T z = 0;
            for (int k = 0; k < dim2; k++) {
                int idx = getindex(i, j, k, dim1, dim2);
                y[idx] = exp(x[idx] - maxv);
                z += y[idx];
            }

            T invz = 1 / z;
            for (int k = 0; k < dim2; k++) {
                int idx = getindex(i, j, k, dim1, dim2);
                y[idx] *= invz;
            }
        }
    }
}

template<typename T>
void softmax_grad(T *y, T *gy, T *gx, int dim1, int dim2, int dim3) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim3; j++) {
            T sum = 0;
            for (int k = 0; k < dim2; k++) {
                int idx = getindex(i, j, k, dim1, dim2);
                sum += gy[idx] * y[idx];
            }

            for (int k = 0; k < dim2; k++) {
                int idx = getindex(i, j, k, dim1, dim2);
                gx[idx] += y[idx] * (gy[idx] - sum);
            }
        }
    }
}

template<typename T>
void logsoftmax(T *x, T *y, int dim1, int dim2, int dim3) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim3; j++) {
            T maxv = 0;
            for (int k = 0; k < dim2; k++) {
                int idx = getindex(i, j, k, dim1, dim2);
                if (k == 0 || x[idx] > maxv) maxv = x[idx];
            }

            T z = 1e-10;
            for (int k = 0; k < dim2; k++) {
                int idx = getindex(i, j, k, dim1, dim2);
                z += exp(x[idx] - maxv);
            }

            T logz = log(z);
            for (int k = 0; k < dim2; k++) {
                int idx = getindex(i, j, k, dim1, dim2);
                y[idx] = x[idx] - maxv - logz;
            }
        }
    }
}

template<typename T>
void logsoftmax_grad(T *y, T *gy, T *gx, int dim1, int dim2, int dim3) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim3; j++) {
            T sum = 0;
            for (int k = 0; k < dim2; k++) {
                int idx = getindex(i, j, k, dim1, dim2);
                sum += gy[idx];
            }

            for (int k = 0; k < dim2; k++) {
                int idx = getindex(i, j, k, dim1, dim2);
                gx[idx] += gy[idx] - exp(y[idx]) * sum;
            }
        }
    }
}

#define SOFTMAX_CAPI(NAME, T, FUNC) \
void NAME(T *x, T *y, int dim1, int dim2, int dim3) { FUNC(x, y, dim1, dim2, dim3); } \
void NAME ## _ ## grad(T *y, T *gy, T *gx, int dim1, int dim2, int dim3) { \
    FUNC ## _ ## grad(y, gy, gx, dim1, dim2, dim3); \
}

extern "C" {
    SOFTMAX_CAPI(softmax_f32, float, softmax)
    SOFTMAX_CAPI(logsoftmax_f32, float, logsoftmax)
}
