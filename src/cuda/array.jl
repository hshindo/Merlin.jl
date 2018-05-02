export CuArray, CuVector, CuMatrix, CuVecOrMat
export curand, curandn

mutable struct CuArray{T,N} <: AbstractCuArray{T,N}
    mb::MemBlock
    dims::NTuple{N,Int}
end

const CuVector{T} = CuArray{T,1}
const CuMatrix{T} = CuArray{T,2}
const CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

function CuArray{T}(dims::NTuple{N,Int}) where {T,N}
    mb = malloc(sizeof(T)*prod(dims))
    CuArray{T,N}(mb, dims)
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
Base.pointer(x::CuArray{T}, index::Int=1) where T = Ptr{T}(x.mb) + sizeof(T)*(index-1)
Base.Array(src::CuArray{T,N}) where {T,N} = copy!(Array{T}(size(src)), src)

function Base.unsafe_wrap(::Type{CuArray}, ptr::Ptr{T}, dims::NTuple{N,Int}) where {T,N}
    mb = MemBlock(ptr, -1, -1)
    CuArray{T,N}(mb, dims)
end

##### indexing #####
function Base.getindex(x::CuArray, I...)
    src = view(x, I...)
    copy!(similar(src), src)
end
function Base.getindex(x::CuArray{T}, index::Int) where T
    dest = copy!(Array{T}(1), x, 1)
    dest[1]
end
function Base.setindex!(y::CuArray{T}, x::CuArray{T}, I...) where T
    copy!(view(y,I...), x)
end

##### reshape #####
Base.reshape(x::CuArray{T}, dims::NTuple{N,Int}) where {T,N} = CuArray{T,N}(x.mb, dims)
Base.reshape{T}(x::CuArray{T}, dims::Int...) = reshape(x, dims)
Base.vec(x::CuArray{T}) where T = ndims(x) == 1 ? x : CuArray(x.mb, (length(x),))
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
function Base.copy!(dest::Array{T}, src::CuArray{T}, n=length(src)) where T
    @apicall :cuMemcpyDtoH (Ptr{Void},Ptr{Void},Csize_t) dest src n*sizeof(T)
    dest
end
function Base.copy!(dest::CuArray{T}, src::Array{T}, n=length(src); stream=C_NULL) where T
    @apicall :cuMemcpyHtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) dest src n*sizeof(T) stream
    dest
end
function Base.copy!(dest::CuArray{T}, src::CuArray{T}, n=length(src); stream=C_NULL) where T
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) dest src n*sizeof(T) stream
    dest
end
function Base.copy!(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int; stream=C_NULL) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,Ptr{Void}) p_dest p_src n*sizeof(T) stream
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
        $k(gdims, bdims, pointer(x), length(x), T(value))
        x
    end
end
Base.fill(::Type{CuArray}, value::T, dims::NTuple) where T = fill!(CuArray{T}(dims), value)
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
