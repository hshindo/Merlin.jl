import Base: broadcast, broadcast!

function broadcasted(::typeof(+), dest::CuArray{T}, srcs::AbstractCuArray{T}...) where T
    for src in srcs
        dest === src && continue
        @assert ndims(dest) >= ndims(src)
        _broadcast!(+, dest, src)
    end
    dest
end

function _broadcast!(::typeof(+), dest::CuArray{T}, src::CuArray{T}) where T
    if length(dest) == length(src)
        BLAS.axpy!(T(1), src, dest)
    else
        CUDNN.addto!(1, src, 1, dest)
    end
    dest
end

@generated function _broadcast!(::typeof(+), dest::CuArray{T,N}, src::AbstractCuArray{T}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void broadcast_add(Array<$Ct,$N> dest, Array<$Ct,$N> src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dest.length()) return;

        int sub[$N];
        dest.ind2sub(sub, idx);
        dest(sub) += src(sub);
    }""")
    quote
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, dest, src)
        dest
    end
end

function broadcast!(::typeof(*), y::CuArray{T}, x1::CuArray{T}, x2::CuArray{T}) where T
    throw("Not implemented yet.")
end
