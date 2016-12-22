import Base.LinAlg.BLAS: axpy!
import Base.broadcast!

abstract AbstractCuArray{T,N}
export AbstractCuArray

function Base.copy!{T,N}(y::AbstractCuArray{T,N}, x::AbstractCuArray{T,N})
    t = ctype(T)
    f = @nvrtc """
    $array_h
    __global__ void f(Array<$t,$N> y, Array<$t,$N> x) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int len_y = y.length();
        if (idx < len_y) {
            if (x.length() == len_y) y(idx) = x(idx);
            else {
                int subs[$N];
                y.idx2sub(idx, subs);
                y(subs) = x(subs);
            }
        }
    } """
    f(y, x, dx=length(y))
    y
end
