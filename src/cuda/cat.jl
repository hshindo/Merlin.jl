function Base.cat(xs::CuArray{T}...; dims::Int) where T
    dim = dims
    length(xs) == 1 && return xs[1]
    N = max(dim, maximum(ndims,xs))
    cumdim = sum(x -> size(x,dim), xs)
    if dim <= N
        ysize = Base.setindex(size(xs[1]), cumdim, dim)
    elseif dim == N+1
        ysize = (size(xs[1])..., cumdim)
    else
        throw("Error.")
    end
    y = similar(xs[1], ysize)

    offset = 0
    for x in xs
        s = size(x, dim)
        I = ntuple(ndims(y)) do i
            i == dim ? (offset+1:offset+s) : Colon()
        end
        copyto!(view(y,I...), x)
        offset += s
    end
    y
end

@generated function concat_binarysearch!(y::CuArray{T,N}, dim::Int, xs::Vector{CuArray{T,N}}) where {T,N}
    Ct = cstring(T)
    f = CuFunction("""
    $Array_h

    __global__ void concat(Array<$Ct,$N> y, int dim, $Ct** xs, int lengthXs, int *cumdims) {
        int idxY = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxY >= y.length()) return;

        int ndIdx[$N];
        y.idx2ndIdx(ndIdx, idxY);

        int left = 0;
        int right = lengthXs;
        while (left < right - 1) {
            int m = (left + right) / 2;
            if (ndIdx[dim] < cumdims[m]) right = m;
            else left = m;
        }

        int xsIdx = left;
        ndIdx[dim] -= cumdims[xsIdx];

        // get element of x
        int idxX = 0;
        int strideX = 1;
        for (int d = 0; d < $N; d++) {
            idxX += ndIdx[d] * strideX;
            if (d == dim) strideX *= cumdims[xsIdx+1] - cumdims[xsIdx];
            else strideX *= y.dims[d];
        }
        y[idxY] = xs[xsIdx][idxX];
    }""")
    quote
        cumdims = Array{Cint}(length(xs)+1)
        cumdims[1] = 0
        for i = 2:length(cumdims)
            cumdims[i] = cumdims[i-1] + size(xs[i-1],dim)
        end
        d_cumdims = CuArray(cumdims)
        d_xs = cubox(xs)

        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, y, dim-1, d_xs.ptr, length(xs), d_cumdims.ptr)
        y
    end
end
