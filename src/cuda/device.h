template<typename T, int N>
struct Array {
    T *data;
    const int dims[N];
    const int strides[N];
public:
    __device__ int length() {
        int n = dims[0];
        for (int i = 1; i < N; i++) n *= dims[i];
        return n;
    }
    __device__ T &operator[](int idx) { return data[idx]; }
    __device__ T &operator()(int idx0, int idx1) {
        int idx = idx0*strides[0] + idx1*strides[1];
        return data[idx];
    }
    __device__ T &operator()(const int ndidxs[N]) {
        int idx = 0;
        for (int i = 0; i < N; i++) {
            if (dims[i] == 1) continue;
            idx += ndidxs[i] * strides[i];
        }
        return data[idx];
    }
    __device__ void ndindex(int ndidxs[N], int idx) {
        ndidxs[0] = 1;
        for (int i = 1; i < N; i++) ndidxs[i] = ndidxs[i-1] * dims[i-1];

        int temp = idx;
        for (int i = N-1; i >= 0; i--) {
            int a = temp / ndidxs[i];
            temp -= a * ndidxs[i];
            ndidxs[i] = a;
        }
        return;
    }
    __device__ T &operator()(int idx) {
        if (N == 1) return data[idx*strides[0]];

        int cumdims[N];
        cumdims[0] = 1;
        for (int i = 1; i < N; i++) cumdims[i] = cumdims[i-1] * dims[i-1];

        int temp = idx;
        int rawidx = 0;
        for (int i = N-1; i >= 0; i--) {
            int a = temp / cumdims[i];
            temp -= a * cumdims[i];
            rawidx += a * strides[i];
        }
        return data[rawidx];
    }
};
