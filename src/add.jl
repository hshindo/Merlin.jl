export addto!, broadcast_addto!

function addto!(dest::Array{T}, doffs::Int, src::Array{T}, soffs::Int, n::Int) where T
    @assert doffs > 0 && soffs > 0
    axpy!(n, T(1), pointer(src,soffs), 1, pointer(dest,doffs), 1)
    dest
end

function addto!(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int) where T
    @assert doffs > 0 && soffs > 0
    axpy!(n, T(1), cupointer(src,soffs), 1, cupointer(dest,doffs), 1)
    dest
end

function addto!(dest::UniArray{T}, src::UniArray{T}) where T
    @assert length(dest) == length(src)
    addto!(dest, 1, src, 1, length(dest))
end

function addto!(dest::Array{T}, src::SubArray{T}) where T
    dest[:] = dest + src
    dest
end
function addto!(dest::SubArray{T}, src::Array{T}) where T
    dest[:] = dest + src
    dest
end

broadcast_addto!(dest::Array{T}, src::Array{T}) where T = dest .+= src
broadcast_addto!(dest::CuArray, src::CuArray) = CUDA.broadcast_addto!(dest, src)

@generated function addto!(dest::CuArray{T}, src::CuSubArray{T,N}) where {T,N}
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
        src.linear && return addto!(dest, 1, src.parent, src.offset+1, length(dest))
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, pointer(dest), src)
        dest
    end
end

@generated function addto!(dest::CuSubArray{T,N}, src::CuArray{T}) where {T,N}
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
        dest.linear && return addto!(dest.parent, dest.offset+1, src, 1, length(src))
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, dest, pointer(src))
        dest
    end
end
