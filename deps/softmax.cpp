#include <algorithm>
#include <math.h>

using namespace std;

float Exp(float x) { return expf(x); }
double Exp(double x) { return exp(x); }
float Log(float x) { return logf(x); }
double Log(double x) { return log(x); }

template<typename T>
void softmax_fw(T *x, int size1, int size2, T *y, bool logsoftmax = false) {
  for (int m2 = 0; m2 < size2; m2++) {

    T x_max = x[m2 * size1];
    for ( int m1 = 1; m1 < size1; m1++) x_max = max(x_max, x[m1 + m2 * size1]);

    T sum = static_cast<T>(0);
    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      y[i] = Exp(x[i]-x_max);
      sum += y[i];
    }

    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      y[i] /= sum;
      if (logsoftmax) y[i] = Log(y[i]);
    }
  }
}

template<typename T>
void softmax_bw(T *gy, T *y, int size1, int size2, T *gx) {
  for (int m2 = 0; m2 < size2; m2++) {

    T sum = static_cast<T>(0);
    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      sum += gy[i] * y[i];
    }

    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      gx[i] = y[i] * (gy[i] - sum);
    }
  }
}

template<typename T>
void logsoftmax_bw(T *gy, T *y, int size1, int size2, T *gx) {
  for (int m2 = 0; m2 < size2; m2++) {

    T sum = static_cast<T>(0);
    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      sum += gy[i];
    }

    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      gx[i] = gy[i] - Exp(y[i]) * sum;
    }
  }
}

extern "C" {
  void softmax_fw_f32(float *x, int size1, int size2, float *y) {
    softmax_fw(x, size1, size2, y);
  }
  void softmax_fw_f64(double *x, int size1, int size2, double *y) {
    softmax_fw(x, size1, size2, y);
  }
  void softmax_bw_f32(float *gy, float *y, int size1, int size2, float *gx) {
    softmax_bw(gy, y, size1, size2, gx);
  }
  void softmax_bw_f64(double *gy, double *y, int size1, int size2, double *gx) {
    softmax_bw(gy, y, size1, size2, gx);
  }
  void logsoftmax_fw_f32(float *x, int size1, int size2, float *y) {
    softmax_fw(x, size1, size2, y, true);
  }
  void logsoftmax_fw_f64(double *x, int size1, int size2, double *y) {
    softmax_fw(x, size1, size2, y, true);
  }
  void logsoftmax_bw_f32(float *gy, float *y, int size1, int size2, float *gx) {
    logsoftmax_bw(gy, y, size1, size2, gx);
  }
  void logsoftmax_bw_f64(double *gy, double *y, int size1, int size2, double *gx) {
    logsoftmax_bw(gy, y, size1, size2, gx);
  }
}
