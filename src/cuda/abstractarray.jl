export AbstractCuArray, AbstractCuVector, AbstractCuMatrx, AbstractCuVecOrMat
abstract type AbstractCuArray{T,N} end

const AbstractCuVector{T} = AbstractCuArray{T,1}
const AbstractCuMatrix{T} = AbstractCuArray{T,2}
const AbstractCuVecOrMat{T} = Union{AbstractCuVector{T},AbstractCuMatrix{T}}

Base.similar(x::AbstractCuArray{T}) where T = CuArray{T}(size(x))
Base.similar(x::AbstractCuArray{T}, dims::NTuple) where T = CuArray{T}(dims)
Base.similar(x::AbstractCuArray{T}, dims::Int...) where T = similar(x, dims)
Base.length(x::AbstractCuArray) = prod(size(x))
Base.isempty(x::AbstractCuArray) = length(x) == 0
Base.ndims(x::AbstractCuArray{T,N}) where {T,N} = N
Base.eltype(x::AbstractCuArray{T}) where T = T
Base.zeros(x::AbstractCuArray{T,N}) where {T,N} = zeros(CuArray{T}, size(x))
Base.ones(x::AbstractCuArray{T}) where T = ones(CuArray{T}, size(x))
Base.copy(src::AbstractCuArray) = copy!(similar(src), src)

function Base.copy!(dest::AbstractCuArray{T}, src::AbstractCuArray{T}) where T
    @assert length(dest) == length(src)
    if iscontigious(dest) && iscontigious(src)
        if isa(dest, CuSubArray)
            dest = unsafe_wrap(CuArray, pointer(dest), size(dest))
        end
        if isa(src, CuSubArray)
            src = unsafe_wrap(CuArray, pointer(src), size(src))
        end
        copy!(dest, src)
    else
        copy!(CuDeviceArray(dest), CuDeviceArray(src))
    end
    dest
end

function add!(dest::AbstractCuArray{T}, src::AbstractCuArray{T}) where T
    @assert length(dest) == length(src)
    if iscontigious(dest) && iscontigious(src)
        if isa(dest, CuSubArray)
            dest = unsafe_wrap(CuArray, pointer(dest), size(dest))
        end
        if isa(src, CuSubArray)
            src = unsafe_wrap(CuArray, pointer(src), size(src))
        end
        BLAS.axpy!(T(1), src, dest)
    else
        add!(CuDeviceArray(dest), CuDeviceArray(src))
    end
    dest
end
