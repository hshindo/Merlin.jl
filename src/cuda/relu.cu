template <typename T>
__device__ void relu_forward(const int n, const T* in, T* out, T negative_slope) {
  CUDA_LOOP(index, n) {
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

extern "C" {
  __global__ void relu_forward_float(float epsilon, float *data, int len) {
    relu_forward(epsilon, data, len);
  }
  __global__ void relu_forward_double(double epsilon, double *data, int len) {
    relu_forward(epsilon, data, len);
  }
}
