#include <algorithm>
#include <stdio.h>

template <typename T>
void window2d_fwd(T *x, T *y, int *size_x, int *params) {
  int x1 = size_x[0], x2 = size_x[1], x3 = size_x[2];
  int w1 = params[0], w2 = params[1];
  int s1 = params[2], s2 = params[3];
  int p1 = params[4], p2 = params[5];
  int n1 = (x1 + 2 * p1 - w1) / s1 + 1;
  int n2 = (x2 + 2 * p2 - w2) / s2 + 1;
  int o = 0;
  for (int i = 0; i < w1 * w2 * x3; i++) {
    int d1 = i % w1;
    int d2 = (i / w1) % w2;
    int d3 = i / (w1 * w2);

    for (int k2 = 0; k2 < n2; k2++) {
      for (int k1 = 0; k1 < n1; k1++) {
        int i1 = k1*s1 - p1 + d1;
        int i2 = k2*s2 - p2 + d2;
        if (i1 >= 0 && i1 < x1 && i2 >= 0 && i2 < x2) {
          y[o] = x[i1 + x1*i2 + x1*x2*d3];
        }
        else y[o] = 0;
        o++;
      }
    }
  }
}

template <typename T>
void window2d_bwd(T *gx, T *gy, int *sizes) {
  int x1 = sizes[0], x2 = sizes[1], x3 = sizes[2];
  int w1 = sizes[3], w2 = sizes[4];
  int s1 = sizes[5], s2 = sizes[6];
  int p1 = sizes[7], p2 = sizes[8];
  int n1 = (x1 + 2 * p1 - w1) / s1 + 1;
  int n2 = (x2 + 2 * p2 - w2) / s2 + 1;
  int o = 0;
  for (int i = 0; i < w1 * w2 * x3; i++) {
    int d1 = i % w1;
    int d2 = (i / w1) % w2;
    int d3 = i / (w1 * w2);

    for (int k2 = 0; k2 < n2; k2++) {
      for (int k1 = 0; k1 < n1; k1++) {
        int i1 = k1*s1 - p1 + d1;
        int i2 = k2*s2 - p2 + d2;
        if (i1 >= 0 && i1 < x1 && i2 >= 0 && i2 < x2) {
          gx[i1 + x1*i2 + x1*x2*d3] += gy[o];
        }
        o++;
      }
    }
  }
}

/*
template <typename T>
void window2d_fwd2(T *x, T *y, int *sizes) {
  int x1 = sizes[0], x2 = sizes[1], x3 = sizes[2];
  int w1 = sizes[3], w2 = sizes[4];
  int s1 = sizes[5], s2 = sizes[6];
  int p1 = sizes[7], p2 = sizes[8];
  int o = 0;
  for (int n2 = 0; n2 < w2; n2++) {
    for (int n1 = 0; n1 < w1; n1++) {
      for (int m3 = 0; m3 < x3; m3++) {
        for (int m2 = -p2+n2; m2 <= x2+p2-w2; m2 += s2) {
          for (int m1 = -p1+n1; m1 <= x1+p1-w1; m1 += s1) {
            if (m1 >= 0 && m1 < x1 && m2 >= 0 && m2 < x2) {
              y[o] = x[m1 + m2*x1 + m3*x1*x2];
            }
            else y[o] = 0;
            o++;
          }
        }
      }
    }
  }
}
*/

extern "C" {
  void window2d_fwd_f32(float *x, float *y, int *size_x, int *params) { window2d_fwd(x, y, size_x, params); }
  //void window2d_fwd_f64(double *x, double *y, int *sizes) { window2d_fwd(x, y, sizes); }
  void window2d_bwd_f32(float *gx, float *gy, int *sizes) { window2d_bwd(gx, gy, sizes); }
  //void window2d_bwd_f64(double *gx, double *gy, int *sizes) { window2d_bwd(gx, gy, sizes); }
}
