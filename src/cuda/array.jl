export CuArray, CuVector, CuMatrix, CuVecOrMat
export curand, curandn, device

mutable struct CuArray{T,N}
    ptr::CuPtr
    dims::NTuple{N,Int}
end

const CuVector{T} = CuArray{T,1}
const CuMatrix{T} = CuArray{T,2}
const CuVecOrMat{T} = Union{CuVector{T},CuMatrix{T}}

function CuArray{T}(dims::NTuple{N,Int}) where T
    ptr = CuPtr(sizeof(T)*prod(dims))
    CuArray{T,N}(ptr, dims)
end
CuArray{T}(dims::Int...) where T = CuArray{T}(dims)
CuArray(x::Array{T,N}) where {T,N} = copy!(CuArray{T}(size(x)), x)

Base.length(x::CuArray) = prod(x.dims)
Base.size(x::CuArray) = x.dims
Base.size(x::CuArray, d::Int) = x.dims[d]
Base.ndims(x::CuArray{T,N}) where {T,N} = N
Base.eltype(x::CuArray{T}) where T = T
function Base.stride(x::CuArray, dim::Int)
    d = 1
    for i = 1:dim-1
        d *= size(x, i)
    end
    d
end
Base.strides(x::CuArray{T,1}) where T = (1,)
Base.strides(x::CuArray{T,2}) where T = (1,size(x,1))
Base.strides(x::CuArray{T,3}) where T = (1,size(x,1),size(x,1)*size(x,2))
Base.strides(x::CuArray{T,4}) where T = (1,size(x,1),size(x,1)*size(x,2),size(x,1)*size(x,2)*size(x,3))
function Base.strides(x::CuArray{T,N}) where {T,N}
    throw("Not implemented yet.")
end

Base.similar(x::CuArray{T}) where T = CuArray{T}(size(x))
Base.similar(x::CuArray{T}, dims::NTuple) where T = CuArray{T}(dims)
Base.similar(x::CuArray{T}, dims::Int...) where T = similar(x, dims)

Base.convert(::Type{Ptr{T}}, x::CuArray) where T = Ptr{T}(x.ptr)
Base.convert(::Type{CUdeviceptr}, x::CuArray) = CUdeviceptr(x.ptr)
Base.unsafe_convert(::Type{Ptr{T}}, x::CuArray) where T = Ptr{T}(x.ptr)
Base.unsafe_convert(::Type{CUdeviceptr}, x::CuArray) = CUdeviceptr(x.ptr)

Base.zeros(x::CuArray{T,N}) where {T,N} = zeros(CuArray{T}, a.dims)
Base.zeros(::Type{CuArray{T}}, dims::Int...) where T = zeros(CuArray{T}, dims)
Base.zeros(::Type{CuArray{T}}, dims::NTuple) where T = fill(CuArray, T(0), dims)

Base.ones(x::CuArray{T}) where T = ones(CuArray{T}, a.dims)
Base.ones(::Type{CuArray{T}}, dims::Int...) where T  = ones(CuArray{T}, dims)
Base.ones(::Type{CuArray{T}}, dims) where T = fill(CuArray, T(1), dims)

function Base.copy!(dest::Array{T}, src::CuArray{T}; stream=C_NULL) where T
    nbytes = length(src) * sizeof(T)
    @apicall :cuMemcpyDtoHAsync (Ptr{Void},Ptr{Void},Csize_t,CuStream_t) dst src nbytes stream
    dest
end
function Base.copy!(dest::CuArray{T}, src::Array{T}; stream=C_NULL) where T
    nbytes = length(src) * sizeof(T)
    @apicall :cuMemcpyHtoDAsync (Ptr{Void},Ptr{Void},Csize_t,CuStream_t) dst src nbytes stream
    dest
end
function Base.copy!(dest::CuArray{T}, src::CuArray{T}; stream=C_NULL) where T
    nbytes = length(src) * sizeof(T)
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,CuStream_t) dst src nbytes stream
    dest
end
function Base.copy!(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int; stream=C_NULL) where T
    _dest = pointer(dest, doffs)
    _src = pointer(src, soffs)
    nbytes = n * sizeof(T)
    @apicall :cuMemcpyDtoDAsync (Ptr{Void},Ptr{Void},Csize_t,CuStream_t) _dst _src nbytes stream
    dest
end

Base.copy(src::CuArray) = copy!(similar(src), src)
Base.pointer(x::CuArray{T}, index::Int=1) where T = Ptr{T}(x) + sizeof(T)*(index-1)
Base.Array(src::CuArray{T,N}) where {T,N} = copy!(Array{T}(size(src)), src)
Base.isempty(x::CuArray) = length(x) == 0
Base.vec(x::CuArray) = ndims(x) == 1 ? x : CuArray(x.ptr, (length(x),))
Base.fill(::Type{CuArray}, value::T, dims::NTuple) where T = fill!(CuArray{T}(dims), value)

@generated function Base.fill!(x::CuArray{T}, value) where T
    Ct = cstring(T)
    f = CuFunction("""
    __global__ void f($Ct *x, int length, $Ct value) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < length) x[idx] = value;
    }""")
    quote
        launch()
        $f(x.ptr, length(x), T(value), dx=length(x))
        x
    end
end

Base.reshape{T,N}(a::CuArray{T}, dims::NTuple{N,Int}) = CuArray{T,N}(a.ptr, dims)
Base.reshape{T}(a::CuArray{T}, dims::Int...) = reshape(a, dims)

Base.getindex(a::CuArray, key...) = copy!(view(a, key...))

function Base.setindex!{T,N}(y::CuArray{T,N}, x::CuArray{T,N}, indexes...)
    if N <= 3
        shift = [0,0,0]
        for i = 1:length(indexes)
            idx = indexes[i]
            if typeof(idx) == Colon
            elseif typeof(idx) <: Range
                # TODO: more range check
                @assert length(idx) == size(x,i)
                shift[i] = start(idx) - 1
            else
                throw("Invalid range: $(idx)")
            end
        end
        shiftcopy!(reshape3(y), reshape3(x), (shift[1],shift[2],shift[3]))
    else
        throw("Not implemented yet.")
    end
    #xx = view(x, key...)
    #copy!(xx, value)
end
function Base.setindex!{T}(x::CuArray{T}, value::T, key::Int)
    throw("Not implemented yet.")
end

function curand(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    # TODO: use curand library
    CuArray(rand(T,dims))
end
curand(::Type{T}, dims::Int...) where T = curand(T, dims)

function curandn(::Type{T}, dims::NTuple{N,Int}) where {T,N}
    # TODO: use curand library
    CuArray(Array{T}(randn(dims)))
end
curandn(::Type{T}, dims::Int...) where T = curandn(T, dims)

getdevice(x::CuArray) = x.ptr.dev

#=
function reshape3(x::CuArray, dim::Int)
    d1, d2, d3 = 1, size(x,dim), 1
    for i = 1:dim-1
        d1 *= size(x,i)
    end
    for i = dim+1:ndims(x)
        d3 *= size(x,i)
    end
    reshape(x, (d1,d2,d3))
end
reshape3{T}(x::CuArray{T,1}) = reshape(x, size(x,1), 1, 1)
reshape3{T}(x::CuArray{T,2}) = reshape(x, size(x,1), size(x,2), 1)
reshape3{T}(x::CuArray{T,3}) = x

@generated function shiftcopy!{T}(dest::CuArray{T,3}, src::CuArray{T,3}, shift::NTuple{3,Int})
    CT = cstring(T)
    f = compile("""
    shiftcopy(Array<$CT,3> dest, Array<$CT,3> src, Dims<3> shift) {
        int idx0 = threadIdx.x + blockIdx.x * blockDim.x;
        int idx1 = threadIdx.y + blockIdx.y * blockDim.y;
        int idx2 = threadIdx.z + blockIdx.z * blockDim.z;
        if (idx0 >= src.dims[0] || idx1 >= src.dims[1] || idx2 >= src.dims[2]) return;
        dest(idx0+shift[0], idx1+shift[1], idx2+shift[2]) = src(idx0, idx1, idx2);
    }""")
    quote
        launch($f, size(src), (dest,src,shift))
        dest
    end
end
=#
