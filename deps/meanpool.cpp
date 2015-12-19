#include <algorithm>
#include <limits>

template <typename T>
void meanpool2d_fwd(const T *input, T *output, int insize1, int insize2,
  int winsize1, int winsize2, int stride1, int stride2, int padsize1, int padsize2) {

  int o = 0;
  for (int m2 = -padsize2; m2 <= insize2 + 2*padsize2 - winsize2; m2 += stride2) {
    for (int m1 = -padsize1; m1 <= insize1 + 2*padsize1 - winsize1; m1 += stride1) {
      T maxval = -std::numeric_limits<T>::max();
      ssize_t maxidx = -1;
      for (int n2 = m2; n2 < m2 + winsize2; n2++) {
        for (int n1 = m1; n1 < m1 + winsize1; n1++) {
          int i = n1 + n2 * insize1;
          T val;
          if (n1 >= 0 && n1 < insize1 && n2 >= 0 && n2 < insize2) val = input[i];
          else val = 0;
          if (val > maxval) {
            maxval = val;
            maxidx = i;
          }
        }
      }
      maxidxs[o] = maxidx;
      output[o] = maxval;
      o++;
    }
  }
}
