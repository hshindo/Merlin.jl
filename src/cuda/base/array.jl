export
    CuArray, CuVector, CuMatrix, CuVecOrMat,
    device

type CuArray{T,N} <: AbstractCuArray{T,N}
    ptr::CuPtr{T}
    dims::NTuple{N,Int}
end

typealias CuVector{T} CuArray{T,1}
typealias CuMatrix{T} CuArray{T,2}
typealias CuVecOrMat{T} Union{CuVector{T},CuMatrix{T}}

(::Type{CuArray{T}}){T,N}(dims::NTuple{N,Int}) = CuArray(alloc(T,prod(dims)), dims)
(::Type{CuArray{T}}){T}(dims::Int...) = CuArray{T}(dims)
CuArray{T,N}(x::Array{T,N}) = copy!(CuArray{T}(size(x)), x)
Base.Array{T,N}(x::CuArray{T,N}) = copy!(Array{T}(size(x)), x)

device(x::CuArray) = x.ptr.dev
Base.pointer{T}(x::CuArray{T}, index::Int=1) = Ptr{T}(x.ptr) + sizeof(T) * (index-1)

Base.length(x::CuArray) = prod(x.dims)
Base.size(x::CuArray) = x.dims
Base.size(x::CuArray, dim::Int) = x.dims[dim]
Base.ndims{T,N}(x::CuArray{T,N}) = N
Base.eltype{T}(x::CuArray{T}) = T
Base.isempty(x::CuArray) = length(x) == 0

Base.strides{T}(x::CuArray{T,1}) = (1,)
Base.strides{T}(x::CuArray{T,2}) = (1,size(x,1))
function Base.strides{T}(x::CuArray{T,3})
    s2 = size(x,1)
    s3 = s2 * size(x,2)
    (1,s2,s3)
end
function Base.strides{T}(x::CuArray{T,4})
    s2 = size(x,1)
    s3 = s2 * size(x,2)
    s4 = s3 * size(x,3)
    (1,s2,s3,s4)
end
function Base.stride{T,N}(x::CuArray{T,N}, dim::Int)
    d = 1
    for i = 1:dim-1
        d *= size(x,i)
    end
    d
end

Base.similar{T,N}(x::CuArray{T}, dims::NTuple{N,Int}) = CuArray{T}(dims)
Base.similar(x::CuArray) = similar(x, size(x))
Base.similar(x::CuArray, dims::Int...) = similar(x, dims)

Base.convert{T}(::Type{Ptr{T}}, x::CuArray) = Ptr{T}(x.ptr)
Base.convert(::Type{CUdeviceptr}, x::CuArray) = CUdeviceptr(x.ptr)
Base.convert{T,N}(::Type{CuArray{T,N}}, x::Array{T,N}) = CuArray(x)
Base.convert{T,N}(::Type{Array{T,N}}, x::CuArray{T,N}) = Array(x)
Base.unsafe_convert{T}(::Type{Ptr{T}}, x::CuArray) = Ptr{T}(x.ptr)
Base.unsafe_convert(::Type{CUdeviceptr}, x::CuArray) = CUdeviceptr(x.ptr)

Base.zeros{T,N}(::Type{CuArray{T}}, dims::NTuple{N,Int}) = fill(CuArray, T(0), dims)
Base.zeros{T,N}(x::CuArray{T,N}) = zeros(CuArray{T}, size(x))
Base.zeros{T}(::Type{CuArray{T}}, dims::Int...) = zeros(CuArray{T}, dims)

Base.ones{T}(x::CuArray{T}) = ones(CuArray{T}, x.dims)
Base.ones{T}(::Type{CuArray{T}}, dims::Int...) = ones(CuArray{T}, dims)
Base.ones{T}(::Type{CuArray{T}}, dims) = fill(CuArray, T(1), dims)

##### copy #####
function cucopy!(f::Function, dest, src, n::Int, stream)
    nbytes = n * sizeof(eltype(src))
    f(dest, src, nbytes, stream)
    dest
end
function Base.copy!{T}(dest::Array{T}, src::CuArray{T}; stream=C_NULL)
    cucopy!(cuMemcpyDtoHAsync, dest, src, length(src), stream)
end
function Base.copy!{T}(dest::CuArray{T}, src::Array{T}; stream=C_NULL)
    cucopy!(cuMemcpyHtoDAsync, dest, src, length(src), stream)
end
function Base.copy!{T}(dest::CuArray{T}, src::CuArray{T}; stream=C_NULL)
    cucopy!(cuMemcpyDtoDAsync, dest, src, length(src), stream)
end
Base.copy(x::CuArray) = copy!(similar(x),x)

function Base.fill!{T,N}(x::CuArray{T,N}, value)
    t = ctype(T)
    f = @nvrtc """
    $array_h
    __global__ void f(Array<$t,$N> x, $t value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < x.length()) {
            x[idx] = value;
        }
    } """
    f(x, T(value), dx=length(x))
    x
end
Base.fill{T}(::Type{CuArray}, value::T, dims::NTuple) = fill!(CuArray{T}(dims), value)

Base.reshape{T,N}(x::CuArray{T}, dims::NTuple{N,Int}) = CuArray{T,N}(x.ptr, dims)
Base.reshape{T}(x::CuArray{T}, dims::Int...) = reshape(x, dims)

Base.getindex(x::CuArray, indexes...) = CuArray(view(x,indexes))

Base.setindex!{T,N}(y::CuArray{T,N}, x::CuArray{T,N}, indexes...) = copy!(view(y,indexes),x)

box(x::CuArray) = box(Interop.Array(x))
