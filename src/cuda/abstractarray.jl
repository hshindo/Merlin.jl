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
        p_dest = pointer(dest)
        p_src = pointer(src)
        BLAS.axpy!(length(dest), T(1), p_src, p_dest)
    else
        add!(CuDeviceArray(dest), CuDeviceArray(src))
    end
    dest
end

function add!(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int) where T
    p_dest = pointer(dest, doffs)
    p_src = pointer(src, soffs)
    BLAS.axpy!(n, T(1), p_dest, 1, p_src, 1)
    dest
end
