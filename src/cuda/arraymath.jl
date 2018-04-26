export CuLinearArray
import Base: copy!, fill!, exp, log, broadcast!, +, -, *, /

const CuLinearArray{T,N} = Union{CuArray{T,N},CuSubArray{T,N,true}}

function copy!(dest::Array{T}, src::CuLinearArray{T}, n=length(src)) where T
    nbytes = n * sizeof(T)
    @apicall :cuMemcpyDtoH (Ptr{Void},Ptr{Void},Csize_t) dest src nbytes
    dest
end
function copy!(dest::CuLinearArray{T}, src::Array{T}, n=length(src); stream=C_NULL) where T
    nbytes = n * sizeof(T)
    @apicall :cuMemcpyHtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) dest src nbytes stream
    dest
end
function copy!(dest::CuLinearArray{T}, src::CuLinearArray{T}, n=length(src); stream=C_NULL) where T
    nbytes = n * sizeof(T)
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) dest src nbytes stream
    dest
end
function copy!(dest::CuLinearArray{T}, doffs::Int, src::CuLinearArray{T}, soffs::Int, n::Int; stream=C_NULL) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    nbytes = n * sizeof(T)
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) p_dest p_src nbytes stream
    dest
end
@generated function copy!(dest::AbstractCuArray{T,N}, src::AbstractCuArray{T,N}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void copy(Array<$Ct,$N> dest, Array<$Ct,$N> src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= src.length()) return;

        int ndIdx[$N];
        dest.idx2ndIdx(ndIdx, idx);
        dest(ndIdx) = src(ndIdx);
    }""")
    quote
        @assert length(dest) == length(src)
        gdims, bdims = cudims(length(src))
        $k(gdims, bdims, dest, src)
        dest
    end
end

@generated function fill!(x::CuLinearArray{T}, value) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void fill($Ct *x, int n, $Ct value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        x[idx] = value;
    }""")
    quote
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, pointer(x), length(x), T(value))
        x
    end
end
@generated function fill!(x::AbstractCuArray{T,N}, value) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void fill(Array<$Ct,$N> x, $Ct value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= x.length()) return;

        x(idx) = value;
    }""")
    quote
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, x, T(value))
        x
    end
end

function +(x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    @assert length(x1) == length(x2)
    y = zeros(x1)
    BLAS.axpy!(T(1), x1, y)
    BLAS.axpy!(T(1), x2, y)
    y
end

function -(x1::CuArray{T,N}, x2::CuArray{T,N}) where {T,N}
    @assert length(x1) == length(x2)
    y = zeros(x1)
    BLAS.axpy!(T(1), x1, y)
    BLAS.axpy!(T(-1), x2, y)
    y
end

function -(x::CuArray{T}) where T
    y = zeros(x)
    BLAS.axpy!(T(-1), x, y)
    y
end

function broadcast!(::typeof(+), dest::CuArray{T}, srcs::AbstractCuArray{T}...) where T
    for src in srcs
        dest === src && continue
        @assert ndims(dest) >= ndims(src)
        _broadcast!(+, dest, src)
    end
    dest
end

function _broadcast!(::typeof(+), dest::CuArray{T}, src::CuLinearArray{T}) where T
    if length(dest) == length(src)
        BLAS.axpy!(T(1), src, dest)
    else
        CUDNN.add!(1, src, 1, dest)
    end
    dest
end

@generated function _broadcast!(::typeof(+), dest::CuArray{T,N}, src::AbstractCuArray{T}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void broadcast_add(Array<$Ct,$N> dest, Array<$Ct,$N> src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dest.length()) return;

        int ndIdx[$N];
        dest.idx2ndIdx(ndIdx, idx);
        dest(ndIdx) = src(ndIdx);
    }""")
    quote
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, dest, src)
        dest
    end
end

*(A::CuMatrix{T}, x::CuVector{T}) where T = BLAS.gemv('N', T(1), A, x)
*(A::CuMatrix{T}, B::CuMatrix{T}) where T = BLAS.gemm('N', 'N', T(1), A, B)

function broadcast!(::typeof(*), y::CuArray{T}, x1::CuArray{T}, x2::CuArray{T}) where T
    throw("Not implemented yet.")
end
