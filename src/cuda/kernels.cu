#define THREADS_PER_BLOCK_X 128
#define THREADS_PER_BLOCK_Y 1
#define THREADS_PER_BLOCK_Z 8

#define LOG_THRESHOLD 1e-20

#define RELU_BOUNDS_AND_INDEX \
  int idx = threadIdx.x + blockIdx.x * blockDim.x; \
  if (idx >= len) \
    return

template <typename T>
__device__ void relu_forward(T epsilon, T *data, int len) {
  RELU_BOUNDS_AND_INDEX;
  data[idx] = max(data[idx], epsilon);
}

template <typename T>
__device__ void relu_backward(T epsilon, T *data, T *gradient, int len) {
  RELU_BOUNDS_AND_INDEX;
  gradient[idx] *= data[idx] > epsilon;
}

extern "C" {
  __global__ void relu_forward_float(float epsilon, float *data, int len) {
    relu_forward(epsilon, data, len);
  }
  __global__ void relu_forward_double(double epsilon, double *data, int len) {
    relu_forward(epsilon, data, len);
  }
  __global__ void relu_backward_float(float epsilon, float *data, float *gradient, int len) {
    relu_backward(epsilon, data, gradient, len);
  }
  __global__ void relu_backward_double(double epsilon, double *data, double *gradient, int len) {
    relu_backward(epsilon, data, gradient, len);
  }
}
