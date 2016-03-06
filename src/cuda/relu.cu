const int CUDA_NUM_THREADS = 512;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) { return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS; }

// CUDA: grid stride looping
#define CUDA_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

template <typename T>
__global__ void relu_forward(T epsilon, T *out, int n) {
	CUDA_LOOP(index, n) {
		out[index] = out[index] > 0 ? out[index] : out[index] * epsilon;
	}
}

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

# relu_forward<float><<<GET_BLOCKS(1), CUDA_NUM_THREADS>>>((float)0.0, raw_ptr, 4);
