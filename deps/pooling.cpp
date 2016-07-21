#include <limits>

template <typename T>
void maxpooling2d(T *x, T *y, int *size_x, int *params, int *maxidxs) {
  int x1 = size_x[0], x2 = size_x[1], x3 = size_x[2];
  int w1 = params[0], w2 = params[1];
  int s1 = params[2], s2 = params[3];
  int p1 = params[4], p2 = params[5];

  int o = 0;
  


  for (int m2 = 0; m2 <= size_x2-w2; m2 += s2) {
    for (int m1 = 0; m1 <= size_x1-w1; m1 += s1) {
      T maxval = std::numeric_limits<T>::min();
      int maxi = -1;
      for (int n2 = m2; n2 < m2 + w2; n2++) {
        for (int n1 = m1; n1 < m1 + w1; n1++) {
          int i = n1 + n2 * size_x1;
          T val;
          if (n1 >= 0 && n1 < size_x1 && n2 >= 0 && n2 < size_x2) val = x[i];
          else val = 0;
          if (val > maxval) {
            maxval = val;
            maxi = i;
          }
        }
      }
      maxind[o] = maxi;
      y[o] = maxval;
      o++;
    }
  }
}

template <typename T>
void maxpooling2d_bwd(const int *maxind, const T *gy, T *gx, int size_y) {
  for (int i = 0; i < size_y; i++) gx[maxind[i]] += gy[i];
}
