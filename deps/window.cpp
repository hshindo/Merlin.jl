#include <algorithm>

template <typename T>
void window2d_fwd(const T *input, T *output, int insize1, int insize2,
  int winsize1, int winsize2, int stride1, int stride2, int padsize1, int padsize2) {

  int o = 0;
  for (int m2 = -padsize2; m2 <= insize2 + 2*padsize2 - winsize2; m2 += stride2) {
    for (int m1 = -padsize1; m1 <= insize1 + 2*padsize1 - winsize1; m1 += stride1) {
      for (int n2 = m2; n2 < m2 + winsize2; n2++) {
        for (int n1 = m1; n1 < m1 + winsize1; n1++) {
          int i = n1 + n2 * insize1;
          if (n1 >= 0 && n1 < insize1 && n2 >= 0 && n2 < insize2) output[o] = input[i];
          else output[o] = 0;
          o++;
        }
      }
    }
  }
}

template <typename T>
void window2d_bwd(const T *gradout, T *gradin, int insize1, int insize2,
  int winsize1, int winsize2, int stride1, int stride2, int padsize1, int padsize2) {

  int o = 0;
  for (int m2 = -padsize2; m2 <= insize2 + 2*padsize2 - winsize2; m2 += stride2) {
    for (int m1 = -padsize1; m1 <= insize1 + 2*padsize1 - winsize1; m1 += stride1) {
      for (int n2 = m2; n2 < m2 + winsize2; n2++) {
        for (int n1 = m1; n1 < m1 + winsize1; n1++) {
          int i = n1 + n2 * insize1;
          if (n1 >= 0 && n1 < insize1 && n2 >= 0 && n2 < insize2) gradin[i] += gradout[o];
          o++;
        }
      }
    }
  }
}

extern "C" {
  void window2d_fwd_f32(const float *input, float *output, int insize1, int insize2,
    int winsize1, int winsize2, int stride1, int stride2, int padsize1, int padsize2) {
    window2d_fwd(input, output, insize1, insize2, winsize1, winsize2, stride1, stride2, padsize1, padsize2);
  }
  void window2d_bwd_f32(const float *gradout, float *gradin, int insize1, int insize2,
    int winsize1, int winsize2, int stride1, int stride2, int padsize1, int padsize2) {
    window2d_bwd(gradout, gradin, insize1, insize2, winsize1, winsize2, stride1, stride2, padsize1, padsize2);
  }
}
