const array_h = """
template<typename T, int N>
struct Array {
    T *data;
    const int dims[N];
    const int strides[N];
    const bool continuous;
public:
    __device__ int length() {
        int n = dims[0];
        for (int i = 1; i < N; i++) n *= dims[i];
        return n;
    }
    __device__ T& operator[](const int idx) { return data[idx]; }
    __device__ T& operator()(int idx0, int idx1) {
        return data[idx0*strides[0] + idx1*strides[1]];
    }
    __device__ T& operator()(int idx0, int idx1, int idx2) {
        return data[idx0*strides[0] + idx1*strides[1] + idx2*strides[2]];
    }
    __device__ T& operator()(int *subs) {
        int idx = 0;
        for (int i = 0; i < N; i++) {
            if (dims[i] > 1) idx += subs[i] * strides[i];
        }
        return data[idx];
    }
    __device__ T& operator()(int idx) {
        if (continuous) return data[idx];
        int cumdims[N];
        cumdims[0] = 1;
        for (int i = 1; i < N; i++) cumdims[i] = cumdims[i-1] * dims[i-1];

        int temp = idx;
        int o = 0;
        for (int i = N-1; i >= 1; i--) {
            int k = temp / cumdims[i];
            o += k * strides[i];
            temp -= k * cumdims[i];
        }
        o += temp * strides[0];
        return data[o];
    }
    __device__ void idx2sub(const int idx, int *subs) {
        int cumdims[N];
        cumdims[0] = 1;
        for (int i = 1; i < N; i++) cumdims[i] = cumdims[i-1] * dims[i-1];

        int temp = idx;
        for (int i = N-1; i >= 1; i--) {
            int k = temp / cumdims[i];
            subs[i] = k;
            temp -= k * cumdims[i];
        }
        subs[0] = temp;
        return;
    }
};
"""
