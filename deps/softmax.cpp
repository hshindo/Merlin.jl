#include "math.h"

inline int getindex(int i, int j, int k, const int *dims) {
  return i + dims[0] * (k + dims[1]*j);
}

template<typename T>
void softmax(T *x, T *y, const int *dims) {
  #pragma omp parallel for
  for (int i = 0; i < dims[0]; i++) {
    for (int j = 0; j < dims[2]; j++) {

      T maxv = x[getindex(i,j,0,dims)];
      for (int k = 0; k < dims[1]; k++) {
        int idx = getindex(i, j, k, dims);
        if (x[idx] > maxv) maxv = x[idx];
      }

      T z = static_cast<T>(0);
      for (int k = 0; k < dims[1]; k++) {
        int idx = getindex(i, j, k, dims);
        y[idx] = exp_approx(x[idx] - maxv);
        z += y[idx];
      }

      T invz = 1 / z;
      for (int k = 0; k < dims[1]; k++) {
        int idx = getindex(i, j, k, dims);
        y[idx] *= invz;
      }
    }
  }
}

template<typename T>
void logsoftmax(T *x, T *y, const int *dims) {
  #pragma omp parallel for
  for (int i = 0; i < dims[0]; i++) {
    for (int j = 0; j < dims[2]; j++) {

      T maxv = x[getindex(i,j,0,dims)];
      for (int k = 0; k < dims[1]; k++) {
        int idx = getindex(i, j, k, dims);
        if (x[idx] > maxv) maxv = x[idx];
      }

      T z = static_cast<T>(0);
      for (int k = 0; k < dims[1]; k++) {
        int idx = getindex(i, j, k, dims);
        z = exp_approx(x[idx] - maxv);
      }

      T logz = log_approx(z);
      for (int k = 0; k < dims[1]; k++) {
        int idx = getindex(i, j, k, dims);
        y[idx] = x[idx] - maxv - logz;
      }
    }
  }
}

#define SOFTMAX_CAPI(NAME, T) \
void NAME ## _ ## T(T *x, T *y, const int *dims) { NAME(x, y, dims); }

extern "C" {
  SOFTMAX_CAPI(softmax, float)
  SOFTMAX_CAPI(softmax, double)
  SOFTMAX_CAPI(logsoftmax, float)
}
