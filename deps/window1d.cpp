template <typename T>
void window1d_fwd(const T *input, T *output, int insize, int winsize, int stride, int padsize) {
  int o = 0;
  for (int a = -padsize; a <= insize + 2*padsize - winsize; a += stride) {
    for (int i = a; i < a + winsize; i++) {
      if (i >= 0 && i < insize) output[o] = input[i];
      else output[o] = 0;
      o++;
    }
  }
}

template <typename T>
void window1d_bwd(const T *input, const T *gradout, T *gradin, int insize, int winsize, int stride, int padsize) {
  for (int i = 0; i < insize; i++) gradin[i] = 0;
  int o = 0;
  //#pragma omp parallel for
  for (int a = -padsize; a <= insize + 2*padsize - winsize; a += stride) {
    for (int i = a; i < a + winsize; i++) {
      if (i >= 0 && i < insize) gradin[i] += gradout[o];
      o++;
    }
  }
}

extern "C" {
  void window1d_fwd_f32(const float *input, float *output, int insize, int winsize, int stride, int padsize) {
    window1d_fwd(input, output, insize, winsize, stride, padsize);
  }
  void window1d_bwd_f32(const float *input, float *gradout, float *gradin, int insize, int winsize, int stride, int padsize) {
    window1d_bwd(input, gradout, gradin, insize, winsize, stride, padsize);
  }
}
