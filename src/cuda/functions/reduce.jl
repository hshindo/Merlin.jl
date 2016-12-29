function sum{T,N}(x::CuArray{T,N}, dim::Int)
    t = ctype(T)
    f = @nvrtc CuArray{T,N} """
    $array_h
    __global__ void f(Array<$t,$N> x, int dim, Array<$t,$N> y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < y.length()) {

        }
    }
    """
    f(length(y), 1, 1, y, x1, x2)
    y
end
