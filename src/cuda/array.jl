export CuArray, CuVector, CuMatrix, CuVecOrMat
export curand, curandn, rawpointer

mutable struct CuArray{T,N} <: AbstractCuArray{T,N}
    ptr::CuPtr{T}
    dims::NTuple{N,Int}
end

const CuVector{T} = CuArray{T,1}
const CuMatrix{T} = CuArray{T,2}
const CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

function CuArray{T}(dims::NTuple{N,Int}) where {T,N}
    ptr = CUDAMalloc()(T, prod(dims))
    CuArray(ptr, dims)
end
CuArray{T}(dims::Int...) where T = CuArray{T}(dims)
CuArray(x::Array{T,N}) where {T,N} = copy!(CuArray{T}(size(x)), x)
CuArray(x::CuArray) = x
iscontigious(x::CuArray) = true

Base.size(x::CuArray) = x.dims
function Base.size(x::CuArray{T,N}, d::Int) where {T,N}
    @assert d > 0
    d <= N ? x.dims[d] : 1
end
Base.strides(x::CuArray{T,1}) where T = (1,)
Base.strides(x::CuArray{T,2}) where T = (1,size(x,1))
Base.strides(x::CuArray{T,3}) where T = (1,size(x,1),size(x,1)*size(x,2))
Base.strides(x::CuArray{T,4}) where T = (1,size(x,1),size(x,1)*size(x,2),size(x,1)*size(x,2)*size(x,3))
function Base.stride(x::CuArray, i::Int)
    d = 1
    for i = 1:i-1
        d *= size(x, i)
    end
    d
end

Base.convert(::Type{Ptr{T}}, x::CuArray) where T = Ptr{T}(pointer(x))
Base.unsafe_convert(::Type{Ptr{T}}, x::CuArray) where T = Ptr{T}(pointer(x))
Base.pointer(x::CuArray, index::Int=1) = index == 1 ? x.ptr : CuPtr(pointer(x.ptr,index),0,x.ptr.dev)
rawpointer(x::CuArray, index::Int=1) = pointer(x.ptr, index)
Base.Array(src::CuArray{T}) where T = copy!(Array{T}(size(src)), src)

##### indexing #####
function Base.getindex(x::CuArray, I...)
    src = view(x, I...)
    copy!(similar(src), src)
end
function Base.getindex(x::CuArray{T}, index::Int) where T
    dest = copy!(Array{T}(1), 1, x, 1, 1)
    dest[1]
end
function Base.setindex!(y::CuArray{T}, x::CuArray{T}, I...) where T
    copy!(view(y,I...), x)
end

##### reshape #####
Base.reshape(x::CuArray{T}, dims::NTuple{N,Int}) where {T,N} = CuArray{T,N}(x.ptr, dims)
Base.reshape{T}(x::CuArray{T}, dims::Int...) = reshape(x, dims)
Base.vec(x::CuArray{T}) where T = ndims(x) == 1 ? x : CuArray(x.ptr, (length(x),))
function Base.squeeze(x::CuArray, dims::Dims)
    for i in dims
        size(x,i) == 1 || throw(ArgumentError("squeezed dims must all be size 1"))
    end
    d = ()
    for i = 1:ndims(x)
        if !in(i, dims)
            d = tuple(d..., size(x,i))
        end
    end
    reshape(x, d)
end
Base.squeeze(x::CuArray, dims::Int...) = squeeze(x, dims)

##### copy #####
Base.copy(src::CuArray) = copy!(similar(src), src)
function Base.copy!(dest::Array{T}, src::CuArray{T}; stream=C_NULL) where T
    @assert length(dest) == length(src)
    copy!(dest, 1, src, 1, length(dest), stream=stream)
end
function Base.copy!(dest::CuArray{T}, src::Array{T}; stream=C_NULL) where T
    @assert length(dest) == length(src)
    copy!(dest, 1, src, 1, length(dest), stream=stream)
end
function Base.copy!(dest::CuArray{T}, src::CuArray{T}; stream=C_NULL) where T
    @assert length(dest) == length(src)
    copy!(dest, 1, src, 1, length(dest), stream=stream)
end

function Base.copy!(dest::Array{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int; stream=C_NULL) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    @apicall :cuMemcpyDtoH (Ptr{Void},Ptr{Void},Csize_t) dest src n*sizeof(T)
    dest
end
function Base.copy!(dest::CuArray{T}, doffs::Int, src::Array{T}, soffs::Int, n::Int; stream=C_NULL) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    @apicall :cuMemcpyHtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) dest src n*sizeof(T) stream
    dest
end
function Base.copy!(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int; stream=C_NULL) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) p_dest p_src n*sizeof(T) stream
    dest
end

function add!(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    BLAS.axpy!(n, T(1), p_src, 1, p_dest, 1)
    dest
end

@generated function Base.fill!(x::CuArray{T}, value) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void fill($Ct *x, int n, $Ct value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        x[idx] = value;
    }""")
    quote
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, rawpointer(x), length(x), T(value))
        x
    end
end
Base.zeros(::Type{CuArray{T}}, dims::Int...) where T = zeros(CuArray{T}, dims)
Base.zeros(::Type{CuArray{T}}, dims::NTuple) where T = fill!(CuArray{T}(dims), 0)
Base.ones(::Type{CuArray{T}}, dims::Int...) where T  = ones(CuArray{T}, dims)
Base.ones(::Type{CuArray{T}}, dims::NTuple) where T = fill!(CuArray{T}(dims), 1)

##### IO #####
Base.show(io::IO, ::Type{CuArray{T,N}}) where {T,N} = print(io, "CuArray{$T,$N}")
function Base.showarray(io::IO, X::CuArray, repr::Bool=true; header=true)
    if repr
        print(io, "CuArray(")
        Base.showarray(io, Array(X), true)
        print(io, ")")
    else
        header && println(io, summary(X), ":")
        Base.showarray(io, Array(X), false, header = false)
    end
end

function curand(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    # TODO: use curand library
    CuArray(rand(T,dims))
end
curand(::Type{T}, dims::Int...) where T = curand(T, dims)
curand(dims::Int...) = curand(Float64, dims)

function curandn(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    # TODO: use curand library
    CuArray(randn(T,dims))
end
curandn(::Type{T}, dims::Int...) where T = curandn(T, dims)
curandn(dims::Int...) = curandn(Float64, dims)

@generated function Base.permutedims(x::CuArray{T,N}, perm::Vector{Int}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void permutedims($Ct *x, int n, $Ct value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        x[idx] = value;
    }""")
    quote
        @assert length(perm) == N
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, rawpointer(x), length(x), T(value))
        x
    end
end
