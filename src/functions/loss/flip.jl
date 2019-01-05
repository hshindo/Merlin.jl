export flip

@generated function flip(x::CuArray{Cint}, K::Int, p::Float64)
    k = Kernel("""
    __global__ void flip(int *y, int *x, int n, float *r, float K, float p) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        float a = 1 - (K-1)/K * p;
        if (r[idx] < a) y[idx] = x[idx];
        else {
            float b = (r[idx]-a) * (K-1) / (1-a);
            int k = int(floorf(b)) + 1;
            if (k < x[idx]) y[idx] = k;
            else y[idx] = k + 1;
        }
    }""")
    quote
        y = similar(x)
        r = curand(Float32, length(y))
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(x), length(y), pointer(r), Float32(K), Float32(p))
        y
    end
end
