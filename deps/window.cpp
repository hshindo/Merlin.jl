#include <algorithm>
#include <stdio.h>

template <typename T>
void window2d(T *x, T *y, int *size_x, int *params) {
  int x1 = size_x[0], x2 = size_x[1], x3 = size_x[2];
  int w1 = params[0], w2 = params[1];
  int s1 = params[2], s2 = params[3];
  int p1 = params[4], p2 = params[5];
  int n1 = (x1 + 2 * p1 - w1) / s1 + 1;
  int n2 = (x2 + 2 * p2 - w2) / s2 + 1;
  int o = 0;
  for (int d3 = 0; d3 < x3; d3++) {
    for (int d2 = 0; d2 < w2; d2++) {
      for (int d1 = 0; d1 < w1; d1++) {
        for (int k2 = 0; k2 < n2; k2++) {
          for (int k1 = 0; k1 < n1; k1++) {
            int i1 = k1*s1 - p1 + d1;
            int i2 = k2*s2 - p2 + d2;
            if (i1 >= 0 && i1 < x1 && i2 >= 0 && i2 < x2) {
              int i = i1 + x1*(i2 + x2*d3);
              y[o] = x[i];
            }
            else y[o] = 0;
            o++;
          }
        }
      }
    }
  }
}

template <typename T>
void window2d_grad(T *gx, T *gy, int *size_x, int *params) {
  int x1 = size_x[0], x2 = size_x[1], x3 = size_x[2];
  int w1 = params[0], w2 = params[1];
  int s1 = params[2], s2 = params[3];
  int p1 = params[4], p2 = params[5];
  int n1 = (x1 + 2 * p1 - w1) / s1 + 1;
  int n2 = (x2 + 2 * p2 - w2) / s2 + 1;
  int o = 0;
  for (int d3 = 0; d3 < x3; d3++) {
    for (int d2 = 0; d2 < w2; d2++) {
      for (int d1 = 0; d1 < w1; d1++) {
        for (int k2 = 0; k2 < n2; k2++) {
          for (int k1 = 0; k1 < n1; k1++) {
            int i1 = k1*s1 - p1 + d1;
            int i2 = k2*s2 - p2 + d2;
            if (i1 >= 0 && i1 < x1 && i2 >= 0 && i2 < x2) {
              int i = i1 + x1*(i2 + x2*d3);
              gx[i] += gy[o];
            }
            o++;
          }
        }
      }
    }
  }
}

#define WINDOW_CAPI(DIM,T) \
void window ## DIM ## _ ## T(T *x, T *y, int *size_x, int *params) { \
  window ## DIM (x, y, size_x, params); \
} \
void window ## DIM ## _ ## grad ## _ ## T(T *gx, T *gy, int *size_x, int *params) { \
  window ## DIM ## _ ## grad(gx, gy, size_x, params); \
}

extern "C" {
  WINDOW_CAPI(2d, float)
  WINDOW_CAPI(2d, double)
}
