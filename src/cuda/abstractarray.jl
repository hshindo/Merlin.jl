export AbstractCuArray, AbstractCuVector, AbstractCuMatrx, AbstractCuVecOrMat
abstract type AbstractCuArray{T,N} end

const AbstractCuVector{T} = AbstractCuArray{T,1}
const AbstractCuMatrix{T} = AbstractCuArray{T,2}
const AbstractCuVecOrMat{T} = Union{AbstractCuVector{T},AbstractCuMatrix{T}}

Base.similar(x::AbstractCuArray{T}) where T = CuArray{T}(size(x))
Base.similar(x::AbstractCuArray{T}, dims::NTuple) where T = CuArray{T}(dims)
Base.similar(x::AbstractCuArray{T}, dims::Int...) where T = similar(x, dims)

Base.length(x::AbstractCuArray) = prod(x.dims)
Base.isempty(x::AbstractCuArray) = length(x) == 0
Base.ndims(x::AbstractCuArray{T,N}) where {T,N} = N
Base.eltype(x::AbstractCuArray{T}) where T = T

Base.size(x::AbstractCuArray) = x.dims
function Base.size(x::AbstractCuArray{T,N}, d::Int) where {T,N}
    @assert d > 0
    d <= N ? x.dims[d] : 1
end

Base.zeros(x::AbstractCuArray{T,N}) where {T,N} = zeros(CuArray{T}, x.dims)
Base.ones(x::AbstractCuArray{T}) where T = ones(CuArray{T}, x.dims)
Base.copy(src::AbstractCuArray) = copy!(similar(src), src)
Base.Array(src::AbstractCuArray{T,N}) where {T,N} = copy!(Array{T}(size(src)), src)
