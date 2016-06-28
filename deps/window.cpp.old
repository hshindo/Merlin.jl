#include <algorithm>

template <typename T>
void window_fwd(const T *x, const int *params, T *y, int size_x1, int size_x2) {
  int w1 = params[0], w2 = params[1];
  int s1 = params[2], s2 = params[3];
  int p1 = params[4], p2 = params[5];
  int o = 0;
  for (int m2 = -p2; m2 <= size_x2 + p2 - w2; m2 += s2) {
    for (int m1 = -p1; m1 <= size_x1 + p1 - w1; m1 += s1) {
      for (int n2 = m2; n2 < m2 + w2; n2++) {
        for (int n1 = m1; n1 < m1 + w1; n1++) {
          int i = n1 + n2 * size_x1;
          if (n1 >= 0 && n1 < size_x1 && n2 >= 0 && n2 < size_x2) y[o] = x[i];
          else y[o] = 0.0;
          o++;
        }
      }
    }
  }
}

template <typename T>
void window_bwd(const int *params, const T *gy, T *gx, int size_x1, int size_x2) {

  int w1 = params[0], w2 = params[1];
  int s1 = params[2], s2 = params[3];
  int p1 = params[4], p2 = params[5];
  int o = 0;
  for (int m2 = -p2; m2 <= size_x2 + p2 - w2; m2 += s2) {
    for (int m1 = -p1; m1 <= size_x1 + p1 - w1; m1 += s1) {
      for (int n2 = m2; n2 < m2 + w2; n2++) {
        for (int n1 = m1; n1 < m1 + w1; n1++) {
          int i = n1 + n2 * size_x1;
          if (n1 >= 0 && n1 < size_x1 && n2 >= 0 && n2 < size_x2) gx[i] += gy[o];
          o++;
        }
      }
    }
  }
}

extern "C" {
  void window_fwd_f32(const float *x, const int *params, float *y, int size_x1, int size_x2) {
    window_fwd(x, params, y, size_x1, size_x2);
  }
  void window_bwd_f32(const int *params, const float *gy, float *gx, int size_x1, int size_x2) {
    window_bwd(params, gy, gx, size_x1, size_x2);
  }
  void window_fwd_f64(const double *x, const int *params, double *y, int size_x1, int size_x2) {
    window_fwd(x, params, y, size_x1, size_x2);
  }
  void window_bwd_f64(const int *params, const double *gy, double *gx, int size_x1, int size_x2) {
    window_bwd(params, gy, gx, size_x1, size_x2);
  }
}
