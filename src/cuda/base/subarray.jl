export CudaSubArray

immutable CudaSubArray{T,N,I} <: AbstractCudaArray{T,N}
    parent::CudaArray{T}
    indexes::I
    offset1::Int
end

typealias CudaSubVector{T} CudaSubArray{T,1}
typealias CudaSubMatrix{T} CudaSubArray{T,2}
typealias CudaSubVecOrMat{T} Union{CudaSubVector{T}, CudaSubMatrix{T}}

Base.Array(x::CudaSubArray) = x.parent[x.indexes...]
Base.SubArray(x::CudaSubArray) = view(Array(x.parent), x.indexes)

function Base.size{T,N,I}(x::CudaSubArray{T,N,I}, dim::Int)
    typeof(x.indexes[dim]) == Colon ? size(x.parent,dim) : length(x.indexes[dim])
end
Base.size{T}(x::CudaSubArray{T,1}) = (size(x,1),)
Base.size{T}(x::CudaSubArray{T,2}) = (size(x,1),size(x,2))
Base.size{T}(x::CudaSubArray{T,3}) = (size(x,1),size(x,2),size(x,3))
Base.size{T}(x::CudaSubArray{T,4}) = (size(x,1),size(x,2),size(x,3),size(x,4))

Base.strides(x::CudaSubArray) = strides(x.parent)
Base.stride(x::CudaSubArray, dim::Int) = stride(x.parent, dim)

Base.length(x::CudaSubArray) = prod(size(x))
Base.similar{T,N}(x::CudaSubArray{T}, dims::NTuple{N,Int}) = CuArray{T}(dims)

function Base.pointer(x::CudaSubArray, index::Int=1)
    index == 1 || throw("Not implemented yet.")
    pointer(x.parent, x.offset1+index)
end

function Base.view{T,N,I<:Tuple}(parent::CudaArray{T,N}, indexes::I)
    @assert N == length(indexes)
    n = N
    offset1 = 0
    for i = 1:length(indexes)
        index = indexes[i]
        typeof(index) == Int && (n -= 1)
        f = typeof(index) == Colon ? 0 : first(index)-1
        offset1 += f * stride(parent,i)
    end
    CudaSubArray{T,n,I}(parent, indexes, offset1)
end
Base.view(parent::CudaArray, indexes::Union{UnitRange{Int},Colon,Int}...) = view(parent,indexes)

box(x::CudaSubArray) = box(Interop.Array(x))
