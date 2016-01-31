#include <algorithm>
#include <arrayfire.h>
#include <stdio.h>
using namespace af;

// array window2d_fwd(void *x, const int *params, void *y, int size_x1, int size_x2) {
//
//   int w1 = params[0], w2 = params[1];
//   int s1 = params[2], s2 = params[3];
//   int p1 = params[4], p2 = params[5];
//   array arr = static_cast<array>(x);
//   // af::array arr = af::array(size_x1, size_x2, x);
//   af_print(arr);
//   array res = unwrap(arr, w1, w2, s1, s2, p1, p2);
//   return res;
// }
//
// void window2d_bwd(const int *params, const void *gy, void *gx, int size_y1, int size_y2) {
//
//   int w1 = params[0], w2 = params[1];
//   int s1 = params[2], s2 = params[3];
//   int p1 = params[4], p2 = params[5];
//   af::array arr = af::array(size_y1, size_y2, gy);
//   af::array res = af::wrap(arr, size_y1, size_y2, w1, w2, s1, s2, p1, p2);
//   gx = &res;
// }

extern "C" {
void window2d_fwd_f32(void *x, const int *params, void **y, int size_x1, int size_x2) {
  int w1 = params[0], w2 = params[1];
  int s1 = params[2], s2 = params[3];
  int p1 = params[4], p2 = params[5];
  void *out;
  af_unwrap(&out, x, w1, w2, s1, s2, p1, p2, true);
  std::swap(*y, out);
}
void window2d_bwd_f32(const int *params, void *gy, void **gx, int size_y1, int size_y2) {
  int w1 = params[0], w2 = params[1];
  int s1 = params[2], s2 = params[3];
  int p1 = params[4], p2 = params[5];
  void *out;
  af_wrap(&out, gy, size_y1, size_y2, w1, w2, s1, s2, p1, p2, true);
  std::swap(*gx, out);
}
}
