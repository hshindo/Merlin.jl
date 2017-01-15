export
    CudaArray, CudaVector, CudaMatrix, CudaVecOrMat,
    device

type CudaArray{T,N} <: AbstractCudaArray{T,N}
    ptr::CudaPtr{T}
    dims::NTuple{N,Int}
end

typealias CudaVector{T} CudaArray{T,1}
typealias CudaMatrix{T} CudaArray{T,2}
typealias CudaVecOrMat{T} Union{CudaVector{T},CudaMatrix{T}}

(::Type{CudaArray{T}}){T,N}(dims::NTuple{N,Int}) = CudaArray(alloc(T,prod(dims)), dims)
(::Type{CudaArray{T}}){T}(dims::Int...) = CudaArray{T}(dims)
CudaArray{T,N}(x::Array{T,N}) = copy!(CudaArray{T}(size(x)), x)
Base.Array{T,N}(x::CudaArray{T,N}) = copy!(Array{T}(size(x)), x)

device(x::CudaArray) = x.ptr.dev
Base.pointer{T}(x::CudaArray{T}, index::Int=1) = Ptr{T}(x.ptr) + sizeof(T) * (index-1)

Base.length(x::CudaArray) = prod(x.dims)
Base.size(x::CudaArray) = x.dims
Base.size(x::CudaArray, dim::Int) = x.dims[dim]
Base.ndims{T,N}(x::CudaArray{T,N}) = N
Base.eltype{T}(x::CudaArray{T}) = T
Base.isempty(x::CudaArray) = length(x) == 0

Base.strides{T}(x::CudaArray{T,1}) = (1,)
Base.strides{T}(x::CudaArray{T,2}) = (1,size(x,1))
function Base.strides{T}(x::CudaArray{T,3})
    s2 = size(x,1)
    s3 = s2 * size(x,2)
    (1,s2,s3)
end
function Base.strides{T}(x::CudaArray{T,4})
    s2 = size(x,1)
    s3 = s2 * size(x,2)
    s4 = s3 * size(x,3)
    (1,s2,s3,s4)
end
function Base.stride{T,N}(x::CudaArray{T,N}, dim::Int)
    d = 1
    for i = 1:dim-1
        d *= size(x,i)
    end
    d
end

Base.similar{T,N}(x::CudaArray{T}, dims::NTuple{N,Int}) = CudaArray{T}(dims)
Base.similar(x::CudaArray) = similar(x, size(x))
Base.similar(x::CudaArray, dims::Int...) = similar(x, dims)

Base.convert{T}(::Type{Ptr{T}}, x::CudaArray{T}) = Ptr{T}(x.ptr)
Base.convert{T,N}(::Type{CudaArray{T,N}}, x::Array{T,N}) = CudaArray(x)
Base.convert{T,N}(::Type{Array{T,N}}, x::CudaArray{T,N}) = Array(x)
Base.unsafe_convert{T}(::Type{Ptr{T}}, x::CudaArray) = Ptr{T}(x.ptr)

Base.zeros{T,N}(::Type{CudaArray{T}}, dims::NTuple{N,Int}) = fill(CudaArray, T(0), dims)
Base.zeros{T,N}(x::CudaArray{T,N}) = zeros(CudaArray{T}, size(x))
Base.zeros{T}(::Type{CudaArray{T}}, dims::Int...) = zeros(CudaArray{T}, dims)

Base.ones{T}(x::CudaArray{T}) = ones(CudaArray{T}, x.dims)
Base.ones{T}(::Type{CudaArray{T}}, dims::Int...) = ones(CudaArray{T}, dims)
Base.ones{T}(::Type{CudaArray{T}}, dims) = fill(CudaArray, T(1), dims)

function cudacopy!(dest, src, count::Int, kind::UInt32, stream)
    nbytes = count * sizeof(eltype(src))
    cudaMemcpyAsync(dest, src, nbytes, kind, stream)
    dest
end
function Base.copy!{T}(dest::Array{T}, src::CudaArray{T}; stream=C_NULL)
    cudacopy!(dest, src, length(src), cudaMemcpyDeviceToHost, stream)
end
function Base.copy!{T}(dest::CudaArray{T}, src::Array{T}; stream=C_NULL)
    cudacopy!(dest, src, length(src), cudaMemcpyHostToDevice, stream)
end
function Base.copy!{T}(dest::CudaArray{T}, src::CudaArray{T}; stream=C_NULL)
    cudacopy!(dest, src, length(src), cudaMemcpyDeviceToDevice, stream)
end
Base.copy(x::CudaArray) = copy!(similar(x), x)

function Base.fill!{T,N}(x::CudaArray{T,N}, value)
    t = ctype(T)
    f = @nvrtc """
    __global__ void f($t *x, int length, $t value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) {
            x[idx] = value;
        }
    } """
    f(x.ptr, length(x), T(value), dx=length(x))
    x
end
Base.fill{T}(::Type{CudaArray}, value::T, dims::NTuple) = fill!(CudaArray{T}(dims), value)

Base.reshape{T,N}(x::CudaArray{T}, dims::NTuple{N,Int}) = CudaArray{T,N}(x.ptr, dims)
Base.reshape{T}(x::CudaArray{T}, dims::Int...) = reshape(x, dims)

function redim{T,N}(x::CudaArray{T,N}, n::Int; pad=0)
    dims = ntuple(n) do i
        1 <= i-pad <= N ? size(x,i-pad) : 1
    end
    reshape(x, dims)
end

Base.getindex(x::CudaArray, indexes...) = CudaArray(view(x,indexes))

Base.setindex!{T,N}(y::CudaArray{T,N}, x::CudaArray{T,N}, indexes...) = copy!(view(y,indexes),x)

box(x::CudaArray) = box(Interop.Array(x))
