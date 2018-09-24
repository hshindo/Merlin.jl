export CuDeviceArray

struct CuDeviceArray{T,N}
    ptr::Ptr{T}
    dims::NTuple{N,Cint}
    strides::NTuple{N,Cint}
end

function CuDeviceArray(x::CuArray, I::Tuple)
    I = Base.to_indices(x, I)
    index = 1
    strides = Base.strides(x)
    for i = 1:length(strides)
        index += strides[i] * (first(I[i])-1)
    end
    dims = length.(I)
    CuDeviceArray(pointer(x,index).ptr, Cint.(dims), Cint.(strides))
end
function CuDeviceArray(x::CuArray)
    CuDeviceArray(pointer(x).ptr, Cint.(size(x)), Cint.(strides(x)))
end

Base.size(x::CuDeviceArray) = Int.(x.dims)
Base.length(x::CuDeviceArray) = Int(prod(x.dims))

function iscontigious(x::CuArray, I::Tuple)
    for i in 1:length(I)-1
        length(I[i]) == size(x,i) || return false
    end
    return true
end

function Base.copyto!(dest::CuArray{T}, I::Tuple, src::CuArray{T}) where T
    I = Base.to_indices(dest, I)
    if iscontigious(dest,I)
        I[end]
    end
    copyto!(CuDeviceArray(dest,I), src)
end

function addto!(dest::CuArray{T}, I::Tuple, src::CuArray{T}) where T
    addto!(CuDeviceArray(dest,I), src)
end

@generated function Base.copyto!(dest::CuArray{T}, src::CuDeviceArray{T,N}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void copyto($Ct *dest, Array<$Ct,$N> src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= src.length()) return;
        dest[idx] = src(idx);
    }""")
    quote
        @assert length(dest) == length(src)
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, pointer(dest), src)
        dest
    end
end

@generated function Base.copyto!(dest::CuDeviceArray{T,N}, src::CuArray{T}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void copyto(Array<$Ct,$N> dest, $Ct *src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dest.length()) return;
        dest(idx) = src[idx];
    }""")
    quote
        @assert length(dest) == length(src)
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, dest, pointer(src))
        dest
    end
end

@generated function addto!(dest::CuArray{T}, src::CuDeviceArray{T,N}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void addto($Ct *dest, Array<$Ct,$N> src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= src.length()) return;
        dest[idx] += src(idx);
    }
    """)
    quote
        @assert length(dest) == length(src)
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, pointer(dest), src)
        dest
    end
end

@generated function addto!(dest::CuDeviceArray{T,N}, src::CuArray{T}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void addto(Array<$Ct,$N> dest, $Ct *src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dest.length()) return;
        dest(idx) += src[idx];
    }
    """)
    quote
        @assert length(dest) == length(src)
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, dest, pointer(src))
        dest
    end
end
