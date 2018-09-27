export CuSubArray, CuSubVector, CuSubMatrix, CuSubVecOrMat

struct CuSubArray{T,N}
    parent::CuArray{T}
    indices::NTuple{N,UnitRange{Int}}
    offset::Int
    linear::Bool
end

const CuSubVector{T} = CuSubArray{T,1}
const CuSubMatrix{T} = CuSubArray{T,2}
const CuSubVecOrMat{T} = Union{CuSubVector{T},CuSubMatrix{T}}

CuArray(x::CuSubArray) = copyto!(similar(x), x)
Base.Array(x::CuSubArray) = Array(CuArray(x))
Base.similar(x::CuSubArray{T}) where T = CuArray{T}(size(x))
Base.size(x::CuSubArray) = map(length, x.indices)
Base.size(x::CuSubArray, i::Int) = length(x.indices[i])
Base.strides(x::CuSubArray) = strides(x.parent)
Base.stride(x::CuSubArray, i::Int) = stride(x.parent, i)
Base.length(x::CuSubArray) = prod(size(x))
Base.pointer(x::CuSubArray, index::Int=1) = pointer(x.parent, x.offset+index)
Base.cconvert(::Type{Ptr{T}}, x::CuSubArray) where T = Base.cconvert(Ptr{T}, pointer(x))

function Base.view(x::CuArray{T,N}, I...) where {T,N}
    @assert length(I) == N
    I = UnitRange.(Base.to_indices(x,I))
    offset = 0
    strides = Base.strides(x)
    for i = 1:length(strides)
        offset += strides[i] * (first(I[i])-1)
    end
    linear = true
    for i = 1:length(I)-1
        length(I[i]) == size(x,i) && continue
        linear = false
        break
    end
    CuSubArray(x, I, offset, linear)
end

##### IO #####
Base.show(io::IO, x::CuSubArray) = print(io, Array(x))

@generated function Base.copyto!(dest::CuArray{T}, src::CuSubArray{T,N}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void copyto($Ct *dest, Array<$Ct,$N> src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= src.length()) return;
        dest[idx] = src(idx);
    }""")
    quote
        @assert length(dest) == length(src)
        src.linear && return copyto!(dest, 1, src.parent, src.offset+1, length(dest))
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, pointer(dest), src)
        dest
    end
end

@generated function Base.copyto!(dest::CuSubArray{T,N}, src::CuArray{T}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void copyto(Array<$Ct,$N> dest, $Ct *src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dest.length()) return;
        dest(idx) = src[idx];
    }""")
    quote
        @assert length(dest) == length(src)
        dest.linear && return copyto!(dest.parent, dest.offset+1, src, 1, length(src))
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, dest, pointer(src))
        dest
    end
end
