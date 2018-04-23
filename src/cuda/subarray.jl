export CuSubArray, CuSubVector, CuSubMatrix, CuSubVecOrMat

type CuSubArray{T,N,L} <: AbstractCuArray{T,N}
    parent::CuArray{T}
    indexes::Tuple
    offset::Int
    dims::NTuple{N,Int}
    strides::NTuple{N,Int}
end

const CuSubVector{T} = CuSubArray{T,1}
const CuSubMatrix{T} = CuSubArray{T,2}
const CuSubVecOrMat{T} = Union{CuSubVector{T},CuSubMatrix{T}}

Base.strides(a::CuSubArray) = a.strides
Base.strides(a::CuSubArray, dim::Int) = a.strides[dim]

Base.convert(::Type{Ptr{T}}, x::CuSubArray) where T = Ptr{T}(pointer(x))
Base.unsafe_convert(::Type{Ptr{T}}, x::CuSubArray) where T = Ptr{T}(pointer(x))

Base.pointer(x::CuSubArray, index::Int=1) = pointer(x.parent, x.offset+index)

function Base.view(x::CuArray{T,N}, indexes...) where {T,N}
    @assert ndims(x) == length(indexes)
    dims = Int[]
    strides = Int[]
    stride = 1
    offset = 0
    L = true # linear index
    for i = 1:length(indexes)
        r = indexes[i]
        if isa(r, Colon)
            if !isempty(dims)
                L = L && (strides[end]*dims[end] == stride)
            end
            push!(dims, size(x,i))
            push!(strides, stride)
        elseif isa(r, Int)
            offset += stride * (r-1)
        elseif isa(r, UnitRange{Int})
            if !isempty(dims)
                L = L && (strides[end]*dims[end] == stride)
            end
            push!(dims, length(r))
            push!(strides, stride)
            offset += stride * (first(r)-1)
        else
            throw("Invalid index: $t.")
        end
        stride *= size(x,i)
    end
    CuSubArray{T,length(dims),L}(x, indexes, offset, tuple(dims...), tuple(strides...))
end

CuArray(x::CuSubArray) = copy!(similar(x), x)

Base.show(io::IO, ::Type{CuSubArray{T,N}}) where {T,N} = print(io, "CuSubArray{$T,$N}")
function Base.showarray(io::IO, X::CuSubArray, repr::Bool=true; header=true)
    if repr
        print(io, "CuSubArray(")
        Base.showarray(io, Array(X), true)
        print(io, ")")
    else
        header && println(io, summary(X), ":")
        Base.showarray(io, Array(X), false, header = false)
    end
end
