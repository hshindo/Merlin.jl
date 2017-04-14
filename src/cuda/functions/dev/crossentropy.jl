import Merlin: crossentropy, ∇crossentropy!

function crossentropy{T}(p::CuVector{Cint}, logq::CuArray{T})
    length(p) == size(logq,2) || throw(DimensionMismatch())

    y = CuArray{T}(1, length(p))
    t = ctype(T)
    f = @compile """
    __global__ void f(int *p, $t *logq, $t *y, int dims[2]) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < col) {
            int i = (p[idx] - 1) + idx * row;
            y[idx] = -logq[i];
        }
    } """
    f(p, logq, y, dx=length(y))
    y
end

function crossentropy2{T}(p::CuVector{Cint}, logq::CuArray{T})
    length(p) == size(logq,2) || throw(DimensionMismatch())

    y = CuArray{T}(1, length(p))
    t = ctype(T)
    f = @compile """
    $array_h
    __global__ void f(Array<int,1> p, Array<$t,2> logq, Array<$t,2> y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < p.length()) {
            y[idx] = -logq(p[idx]-1, idx);
        }
    } """
    f(p, logq, y, dx=length(y))
    y
end

function ∇crossentropy!{T}(gy::CuMatrix{T}, p::CuVector{Cint}, logq::CuMatrix{T}, gx::CuMatrix{T})
    length(p) == size(logq,2) || throw(DimensionMismatch())

    y = CuArray{T}(1, length(p))
    t = ctype(T)
    f = @compile """
    $array_h
    __global__ void f(Array<int,1> p, Array<$t,2> logq, Array<$t,2> y) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < p.length()) {
            y[idx] = -logq(p[idx]-1, idx);
        }
    } """
    f(p, logq, y, dx=length(y))
    y
end
