export CuSubArray, CuSubVector, CuSubMatrix, CuSubVecOrMat

struct CuSubArray{T,N} <: AbstractCuArray{T,N}
    parent::CuArray{T}
    indices::Tuple
    offset::Int
    contigious::Bool
end

const CuSubVector{T} = CuSubArray{T,1}
const CuSubMatrix{T} = CuSubArray{T,2}
const CuSubVecOrMat{T} = Union{CuSubVector{T},CuSubMatrix{T}}

CuArray(x::CuSubArray) = copyto!(similar(x), x)
Base.Array(x::CuSubArray) = Array(CuArray(x))
Base.size(x::CuSubArray) = map(length, x.indices)
Base.size(x::CuSubArray, i::Int) = length(x.indices[i])
Base.strides(x::CuSubArray) = strides(x.parent)
Base.stride(x::CuSubArray, dim::Int) = stride(x.parent, dim)
Base.convert(::Type{Ptr{T}}, x::CuSubArray) where T = Ptr{T}(pointer(x))
Base.unsafe_convert(::Type{Ptr{T}}, x::CuSubArray) where T = Ptr{T}(pointer(x))
Base.pointer(x::CuSubArray, index::Int=1) = pointer(x.parent, x.offset+index)
iscontigious(x::CuSubArray) = x.contigious

function Base.view(x::CuArray{T,N}, I...) where {T,N}
    @assert length(I) == N
    I = Base.to_indices(x, I)
    offset = sub2ind(size(x), map(first,I)...) - 1
    function iscontigious(x::CuArray, I::Tuple)
        for i = 1:length(I)-1
            length(I[i]) == size(x,i) && continue
            length(I[i+1]) == 1 && continue
            return false
        end
        true
    end
    CuSubArray{T,N}(x, I, offset, iscontigious(x,I))
end

Base.show(io::IO, ::Type{CuSubArray{T,N}}) where {T,N} = print(io, "CuSubArray{$T,$N}")
Base.print_array(io::IO, X::CuSubArray) = Base.showarray(io, Array(X))
