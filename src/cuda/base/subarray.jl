export
    CuSubArray

immutable CuSubArray{T,N,I} <: AbstractCuArray{T,N}
    parent::CuArray{T}
    indexes::I
    offset1::Int
end

typealias CuSubVector{T} CuArray{T,1}
typealias CuSubMatrix{T} CuArray{T,2}
typealias CuSubVecOrMat{T} Union{CuSubVector{T}, CuSubMatrix{T}}

Base.Array(x::CuSubArray) = x.parent[x.indexes...]
Base.SubArray(x::CuSubArray) = view(Array(x.parent), x.indexes)

function Base.size{T,N,I}(x::CuSubArray{T,N,I}, dim::Int)
    typeof(x.indexes[dim]) == Colon ? size(x.parent,dim) : length(x.indexes[dim])
end
Base.size{T}(x::CuSubArray{T,1}) = (size(x,1),)
Base.size{T}(x::CuSubArray{T,2}) = (size(x,1),size(x,2))
Base.size{T}(x::CuSubArray{T,3}) = (size(x,1),size(x,2),size(x,3))
Base.size{T}(x::CuSubArray{T,4}) = (size(x,1),size(x,2),size(x,3),size(x,4))

Base.strides(x::CuSubArray) = strides(x.parent)
Base.stride(x::CuSubArray, dim::Int) = stride(x.parent, dim)

Base.length(x::CuSubArray) = prod(size(x))
Base.similar{T,N}(x::CuSubArray{T}, dims::NTuple{N,Int}) = CuArray{T}(dims)

function Base.pointer(x::CuSubArray, index::Int=1)
    index == 1 || throw("Not implemented yet.")
    pointer(x.parent, x.offset1+index)
end

function Base.view{T,N,I<:Tuple}(parent::CuArray{T,N}, indexes::I)
    @assert N == length(indexes)
    n = N
    offset1 = 0
    for i = 1:length(indexes)
        index = indexes[i]
        typeof(index) == Int && (n -= 1)
        f = typeof(index) == Colon ? 0 : first(index)-1
        offset1 += f * stride(parent,i)
    end
    CuSubArray{T,n,I}(parent, indexes, offset1)
end
Base.view(parent::CuArray, indexes::Union{UnitRange{Int},Colon,Int}...) = view(parent,indexes)

box(x::CuSubArray) = box(Interop.Array(x))
