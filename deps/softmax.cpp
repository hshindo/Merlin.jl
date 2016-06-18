#include <algorithm>
#include <math.h>

/* Workaround a lack of optimization in gcc */
float exp_cst1 = 2139095040.f;
float exp_cst2 = 0.f;

/* Relative error bounded by 1e-5 for normalized outputs
   Returns invalid outputs for nan inputs
   Continuous error */
inline float expapprox(float val) {
  union { int i; float f; } xu, xu2;
  float val2, val3, val4, b;
  int val4i;
  val2 = 12102203.1615614f*val + 1065353216.f;
  val3 = val2 < exp_cst1 ? val2 : exp_cst1;
  val4 = val3 > exp_cst2 ? val3 : exp_cst2;
  val4i = (int) val4;
  xu.i = val4i & 0x7F800000;
  xu2.i = (val4i & 0x7FFFFF) | 0x3F800000;
  b = xu2.f;

  /* Generated in Sollya with:
     > f=remez(1-x*exp(-(x-1)*log(2)),
               [|1,(x-1)*(x-2), (x-1)*(x-2)*x, (x-1)*(x-2)*x*x|],
               [1,2], exp(-(x-1)*log(2)));
     > plot(exp((x-1)*log(2))/(f+x)-1, [1,2]);
     > f+x;
  */
  return
    xu.f * (0.510397365625862338668154f + b *
            (0.310670891004095530771135f + b *
             (0.168143436463395944830000f + b *
              (-2.88093587581985443087955e-3f + b *
               1.3671023382430374383648148e-2f))));
}

template<typename T>
void softmax_fw(T *x, int size1, int size2, T *y) {
  for (int m2 = 0; m2 < size2; m2++) {
    T x_max = x[m2*size1];
    for (int m1 = 1; m1 < size1; m1++) x_max = std::max(x_max, x[m1 + m2*size1]);

    T z = static_cast<T>(0);
    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      y[i] = expapprox(x[i] - x_max);
      z += y[i];
    }
    T invz = 1 / z;
    for (int m1 = 0; m1 < size1; m1++) y[m1 + m2*size1] *= invz;
  }
}

template<typename T>
void logsoftmax_fw(T *x, int size1, int size2, T *y) {
  for (int m2 = 0; m2 < size2; m2++) {
    T x_max = x[m2*size1];
    for (int m1 = 1; m1 < size1; m1++) x_max = std::max(x_max, x[m1 + m2*size1]);

    T z = static_cast<T>(0);
    for (int m1 = 0; m1 < size1; m1++) z += exp(x[m1 + m2*size1] - x_max);

    T logz = log(z);
    for (int m1 = 0; m1 < size1; m1++) {
      int i = m1 + m2 * size1;
      y[i] = x[i] - x_max - logz;
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
      gx[i] = gy[i] - exp(y[i]) * sum;
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
    logsoftmax_fw(x, size1, size2, y);
  }
  void logsoftmax_fw_f64(double *x, int size1, int size2, double *y) {
    logsoftmax_fw(x, size1, size2, y);
  }
  void logsoftmax_bw_f32(float *gy, float *y, int size1, int size2, float *gx) {
    logsoftmax_bw(gy, y, size1, size2, gx);
  }
  void logsoftmax_bw_f64(double *gy, double *y, int size1, int size2, double *gx) {
    logsoftmax_bw(gy, y, size1, size2, gx);
  }
}
