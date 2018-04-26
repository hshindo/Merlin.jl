template<typename T, int N>
struct Array {
    T *data;
    const int dims[N];
public:
    __device__ int length() {
        int n = dims[0];
        for (int i = 1; i < N; i++) n *= dims[i];
        return n;
    }
    __device__ T &operator[](int idx) { return data[idx]; }
    __device__ T &operator()(int idx) { return data[idx]; }
    __device__ T &operator()(const int sub[N]) {
        int idx = 0;
        int stride = 1;
        for (int i = 0; i < N; i++) {
            if (dims[i] > 1) idx += sub[i] * stride;
            stride *= dims[i];
        }
        return data[idx];
    }
    __device__ T &operator()(const int sub[N], const int offset[N]) {
        int idx = 0;
        int stride = 1;
        for (int i = 0; i < N; i++) {
            if (dims[i] > 1) idx += (sub[i]+offset[i]) * stride;
            stride *= dims[i];
        }
        return data[idx];
    }
    __device__ void ind2sub(const int sub[N], int ind) { ind2sub(sub, ind, dims); }
};

template<int N>
__device__ void ind2sub(int sub[N], int ind, const int dims[N]) {
    sub[0] = 1;
    for (int i = 1; i < N; i++) sub[i] = sub[i-1] * dims[i-1];

    int temp = ind;
    for (int i = N-1; i >= 0; i--) {
        int a = temp / sub[i];
        temp -= a * sub[i];
        sub[i] = a;
    }
    return;
}
