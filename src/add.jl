export addto!, broadcast_addto!

function addto!(dest::Array{T}, doffs::Int, src::Array{T}, soffs::Int, n::Int) where T
    @assert doffs > 0 && soffs > 0
    axpy!(n, T(1), pointer(src,soffs), 1, pointer(dest,doffs), 1)
    dest
end

function addto!(dest::Array{T}, src::Array{T}) where T
    @assert length(dest) == length(src)
    addto!(dest, 1, src, 1, length(dest))
end

function addto!(dest::CuArray{T}, src::CuArray{T}) where T
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
function addto!(β, dest::Array, α, src::Array)
    dest[:] = α*src + β*dest
    dest
end

function broadcast_addto!(β, dest::Array{T}, α, src::Array{T}) where T
    dest[:] = α*src .+ β*dest
    dest
end
broadcast_addto!(dest::Array, src::Array) = broadcast_addto!(1, dest, 1, src)

function addto!(dest::CuArray{T}, doffs::Int, src::CuArray{T}, soffs::Int, n::Int) where T
    @assert doffs > 0 && soffs > 0
    axpy!(n, T(1), pointer(src,soffs), 1, pointer(dest,doffs), 1)
    dest
end
addto!(β, dest::CuArray, α, src::CuArray) = CUDNN.add!(α, src, β, dest)

@generated function broadcast_addto!(β, dest::CuArray{T,N}, α, src::CuArray{T}) where {T,N}
    # CUDNN.add!(α, src, β, dest)
    Ct = cstring(T)
    k = Kernel("""
    __global__ void addto($Ct beta, Array<$Ct,$N> dest, $Ct alpha, Array<$Ct,$N> src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dest.length()) return;

        int ndidxs[$N];
        dest.ndindex(ndidxs, idx);
        dest[idx] = beta * dest[idx] + alpha * src(ndidxs);
    }
    """)
    quote
        @assert length(dest) >= length(src)
        @assert N >= ndims(src)
        dims = size(src)
        for _ = 1:N-ndims(src)
            dims = (dims..., 1)
        end
        isempty(dims) || (src = reshape(src,dims))
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, T(β), dest, T(α), src)
        dest
    end
end
broadcast_addto!(dest::CuArray, src::CuArray) = broadcast_addto!(1, dest, 1, src)

@generated function addto!(dest::CuArray{T}, value::T) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void addto($Ct *dest, $Ct value, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        dest[idx] += value;
    }
    """)
    quote
        gdims, bdims = cudims(length(dest))
        $k(gdims, bdims, pointer(dest), value, length(dest))
        dest
    end
end

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
        int didx = dest.rawindex(idx);
        $Ct d = dest[didx];
        dest[didx] = d + src[idx];
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
