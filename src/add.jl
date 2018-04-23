function add!(dest::AbstractArray{T}, src::AbstractArray{T}) where T
    @assert length(dest) == length(src)
    broadcast!(+, dest, dest, src)
end

function add!(dest::CuLinearArray{T}, src::CuLinearArray{T}) where T
    @assert length(dest) == length(src)
    BLAS.axpy!(T(1), src, dest)
    dest
end

@generated function add!(dest::AbstractCuArray{T,N}, src::AbstractCuArray{T,N}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void add(Array<$Ct,$N> dest, Array<$Ct,$N> src) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= src.length()) return;

        int ndIdx[$N];
        dest.idx2ndIdx(ndIdx, idx);
        dest(ndIdx) += src(ndIdx);
    }
    """)
    quote
        @assert length(dest) == length(src)
        gdims, bdims = cudims(length(src))
        $k(gdims, bdims, dest, src)
        dest
    end
end
