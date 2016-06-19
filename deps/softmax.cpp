#include <algorithm>
#include <math.h>
#include "math.cpp"

template<typename T>
void softmax_fw(T *x, T *y, int size1, int size2) {
  for (int m2 = 0; m2 < size2; m2++) {
    T x_max = x[m2*size1];
    for (int m1 = 1; m1 < size1; m1++) x_max = std::max(x_max, x[m1 + m2*size1]);

    T z = static_cast<T>(0);
    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2*size1;
      y[i] = exp_approx(x[i] - x_max);
      z += y[i];
    }
    T invz = 1 / z;
    for (int m1 = 0; m1 < size1; m1++) y[m1 + m2*size1] *= invz;
  }
}

template<typename T>
void logsoftmax_fw(T *x, T *y, int size1, int size2) {
  for (int m2 = 0; m2 < size2; m2++) {
    T x_max = x[m2*size1];
    for (int m1 = 1; m1 < size1; m1++) x_max = std::max(x_max, x[m1 + m2*size1]);

    T z = static_cast<T>(0);
    for (int m1 = 0; m1 < size1; m1++) z += exp_approx(x[m1 + m2*size1] - x_max);

    T logz = log(z);
    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      y[i] = x[i] - x_max - logz;
    }
  }
}

template<typename T>
void softmax_bw(T *gx, T *y, T *gy, int size1, int size2) {
  for (int m2 = 0; m2 < size2; m2++) {
    T sum = static_cast<T>(0);
    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      sum += gy[i] * y[i];
    }

    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      gx[i] += y[i] * (gy[i] - sum);
    }
  }
}

template<typename T>
void logsoftmax_bw(T *gx, T *y, T *gy, int size1, int size2) {
  for (int m2 = 0; m2 < size2; m2++) {
    T sum = static_cast<T>(0);
    for (int m1 = 0; m1 < size1; m1++) sum += gy[m1 + m2*size1];

    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      gx[i] += gy[i] - exp(y[i]) * sum;
    }
  }
}

extern "C" {
  void softmax_fw_f32(float *x, float *y, int size1, int size2) {
    softmax_fw(x, y, size1, size2);
  }
  void softmax_fw_f64(double *x, double *y, int size1, int size2) {
    softmax_fw(x, y, size1, size2);
  }

  void softmax_bw_f32(float *gx, float *y, float *gy, int size1, int size2) {
    softmax_bw(gx, y, gy, size1, size2);
  }
  void softmax_bw_f64(double *gx, double *y, double *gy, int size1, int size2) {
    softmax_bw(gx, y, gy, size1, size2);
  }

  void logsoftmax_fw_f32(float *x, float *y, int size1, int size2) {
    logsoftmax_fw(x, y, size1, size2);
  }
  void logsoftmax_fw_f64(double *x, double *y, int size1, int size2) {
    logsoftmax_fw(x, y, size1, size2);
  }

  void logsoftmax_bw_f32(float *gx, float *y, float *gy, int size1, int size2) {
    logsoftmax_bw(gx, y, gy, size1, size2);
  }
  void logsoftmax_bw_f64(double *gx, double *y, double *gy, int size1, int size2) {
    logsoftmax_bw(gx, y, gy, size1, size2);
  }
}
