import Base: +, -, *, /
import Base.Broadcast: broadcasted

function +(x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    @assert length(x1) == length(x2)
    y = fill!(similar(x1), 0)
    axpy!(T(1), x1, y)
    axpy!(T(1), x2, y)
    y
end

function -(x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    @assert length(x1) == length(x2)
    y = fill!(similar(x1), 0)
    axpy!(T(1), x1, y)
    axpy!(T(-1), x2, y)
    y
end

function -(x::CuArray{T}) where T
    y = fill!(similar(x1), 0)
    axpy!(T(-1), x, y)
    y
end

*(A::CuMatrix{T}, x::CuVector{T}) where T = gemv('N', T(1), A, x)
*(A::CuMatrix{T}, B::CuMatrix{T}) where T = gemm('N', 'N', T(1), A, B)

function broadcasted(::typeof(+), x1::CuArray{T}, x2::CuArray{T}) where T
    x1, x2 = length(x1) >= length(x2) ? (x1,x2) : (x2,x1)
    y = copy(x1)
    CUDNN.add!(1, x2, 1, y)
    y
end

function broadcasted(::typeof(-), x1::CuArray{T}, x2::CuArray{T}) where T
    x1, x2 = length(x1) >= length(x2) ? (x1,x2) : (x2,x1)
    y = copy(x1)
    CUDNN.add!(-1, x2, 1, y)
    y
end

function broadcast_addto!(dest::CuArray{T}, src::CuArray{T}) where T
    CUDNN.add!(1, src, 1, dest)
end

function broadcasted(::typeof(*), x1::CuArray{T}, x2::CuArray{T}) where T
    elemtimes(x1, x2)
end

@generated function elemtimes(x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void elemtimes($Ct *y, Array<$Ct,$N> x1, Array<$Ct,$N> x2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= x1.length()) return;
        int ndidxs[$N];
        x1.ndindex(ndidxs, idx);
        y[idx] = x1[idx] * x2(ndidxs);
    }
    """)
    quote
        length(x1) < length(x2) && return elemtimes(x2, x1)
        y = similar(x1)
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), x1, x2)
        y
    end
end
