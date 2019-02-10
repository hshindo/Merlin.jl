export CuArray, CuVector, CuMatrix, CuVecOrMat
export cupointer, curand, curandn, cuzeros, cuones

mutable struct CuArray{T,N} <: AbstractArray{T,N}
    ptr::CuPtr{T}
    dims::Dims{N}
end

const CuVector{T} = CuArray{T,1}
const CuMatrix{T} = CuArray{T,2}
const CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

function CuArray{T}(dims::Dims{N}) where {T,N}
    ptr = MemPoolMalloc()(T, dims)
    CuArray(ptr, dims)
end
CuArray{T}(dims::Int...) where T = CuArray{T}(dims)
CuArray(x::Array{T,N}) where {T,N} = copyto!(CuArray{T}(size(x)), x)

### AbstractArray interface
getdevice(x::CuArray) = x.ptr.dev
Base.size(x::CuArray) = x.dims
Base.size(x::CuArray, i::Int) = i <= ndims(x) ? x.dims[i] : 1
Base.getindex(x::CuArray, i::Int) = throw("getindex $i")
Base.getindex(x::CuArray, I...) = CuArray(view(x,I...))
Base.setindex!(y::CuArray, v::Number, i::Int) = throw("setindex!")
Base.setindex!(y::CuArray{T}, x::CuArray{T}, I...) where T = copyto!(view(y,I), x)
Base.IndexStyle(::Type{<:CuArray}) = IndexLinear()
Base.length(x::CuArray) = prod(x.dims)
Base.similar(x::CuArray, ::Type{T}, dims::Dims{N}) where {T,N} = CuArray{T}(dims)
Base.similar(::Type{<:CuArray}, ::Type{T}, dims::Dims) where T = CuArray{T}(dims)
Base.ndims(x::CuArray{T,N}) where {T,N} = N
Base.eltype(x::CuArray{T,N}) where {T,N} = T
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

cuzeros(::Type{T}, dims::Dims) where T = fill!(CuArray{T}(dims), 0)
cuones(::Type{T}, dims::Dims) where T = fill!(CuArray{T}(dims), 1)
Base.zero(x::CuArray{T}, dims::Dims=size(x)) where T = cuzeros(T, dims)
Base.one(x::CuArray{T}, dims::Dims=size(x)) where T = cuones(T, dims)

function Base.pointer(x::CuArray{T}, index::Int=1) where T
    @assert index > 0
    p = x.ptr.ptr + sizeof(T)*(index-1)
    CuPtr(p, getdevice(x), -1)
end
Base.cconvert(::Type{Ptr{T}}, x::CuArray) where T = Base.cconvert(Ptr{T}, pointer(x))
Base.Array(x::CuArray{T}) where T = copyto!(Array{T}(undef,size(x)), x)
#Base.unsafe_wrap(::Type{A}, ptr::Ptr{T}, dims) where {A<:CuArray,T} = CuArray(ptr, dims)

##### reshape #####
function Base.reshape(x::CuArray{T}, dims::Dims{N}) where {T,N}
    @assert prod(dims) == length(x)
    CuArray(x.ptr, dims)
end
Base.reshape(x::CuArray{T}, dims::Int...) where T = reshape(x, dims)
Base.vec(x::CuArray{T}) where T = ndims(x) == 1 ? x : reshape(x,length(x))
function Base.dropdims(x::CuArray; dims)
    isa(dims,Int) && (dims = (dims,))
    for i in dims
        size(x,i) == 1 || throw(ArgumentError("dropdims must all be size 1"))
    end
    d = ()
    for i = 1:ndims(x)
        if !in(i, dims)
            d = tuple(d..., size(x,i))
        end
    end
    reshape(x, d)
end

##### copy #####
Base.copy(src::CuArray) = copyto!(similar(src), src)
function Base.copyto!(dest::Array{T}, src::CuArray{T}; stream=C_NULL) where T
    @assert length(dest) == length(src)
    copyto!(dest, 1, src, 1, length(dest), stream=stream)
end
function Base.copyto!(dest::CuArray{T}, src::Array{T}; stream=C_NULL) where T
    @assert length(dest) == length(src)
    copyto!(dest, 1, src, 1, length(dest), stream=stream)
end
function Base.copyto!(dest::CuArray{T}, src::CuArray{T}; stream=C_NULL) where T
    @assert length(dest) == length(src)
    copyto!(dest, 1, src, 1, length(dest), stream=stream)
end
function Base.copyto!(dest::Array{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int; stream=C_NULL) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    @apicall :cuMemcpyDtoH (Ptr{Cvoid},Ptr{Cvoid},Csize_t) p_dest p_src n*sizeof(T)
    dest
end
function Base.copyto!(dest::CuArray{T}, doffs::Int, src::Array{T}, soffs::Int, n::Int; stream=C_NULL) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    @apicall :cuMemcpyHtoDAsync (Ptr{Cvoid},Ptr{Cvoid},Csize_t,Ptr{Cvoid}) p_dest p_src n*sizeof(T) stream
    dest
end
function Base.copyto!(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int; stream=C_NULL) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    @apicall :cuMemcpyDtoDAsync (Ptr{Cvoid},Ptr{Cvoid},Csize_t,Ptr{Cvoid}) p_dest p_src n*sizeof(T) stream
    dest
end

function Base.fill!(x::CuArray{T}, value; stream=C_NULL) where T
    if sizeof(T) == 4
        ui = reinterpret(Cuint, T(value))
        @apicall :cuMemsetD32Async (Ptr{Cvoid},Cuint,Csize_t,Ptr{Cvoid}) x ui length(x) stream
    elseif sizeof(T) == 2
        us = reinterpret(Cushort, T(value))
        @apicall :cuMemsetD16Async (Ptr{Cvoid},Cushort,Csize_t,Ptr{Cvoid}) x us length(x) stream
    elseif sizeof(T) == 1
        uc = reinterpret(Cuchar, T(value))
        @apicall :cuMemsetD8Async (Ptr{Cvoid},Cuchar,Csize_t,Ptr{Cvoid}) x us length(x) stream
    else
        throw("Not supported.")
    end
    x
end

##### IO #####
Base.show(io::IO, x::CuArray) = print(io, Array(x))
Base.display(x::CuArray) = display(Array(x))

#####
curand(::Type{T}, dims::Dims{N}) where {T,N} = CURAND.curand(T, dims)
curand(::Type{T}, dims::Int...) where T = curand(T, dims)
curand(dims::Int...) = curand(Float64, dims)
curandn(::Type{T}, dims::Dims{N}) where {T,N} = CURAND.curandn(T, dims)
curandn(::Type{T}, dims::Int...) where T = curandn(T, dims)
curandn(dims::Int...) = curandn(Float64, dims)

function Base.repeat(x::CuArray{T,N}, counts::Int...) where {T,N}
    @assert N <= length(counts)
    if N < length(counts)
        x = reshape(x, size(x)..., ntuple(_ -> 1, length(counts)-N)...)
    end
    repeat_kernel(x, counts)
end
