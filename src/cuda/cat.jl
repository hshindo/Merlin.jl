@generated function cat_cuda(xs::Vector{CuArray{T,N}}, dim::Int) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void cat($Ct *y, $Ct *x, int *cumsizeY, int *cumsizeX) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batchI = idx / m;
        idx = idx - batchI * m;
        int sizeY = cumsizeY[batchI+1] - cumsizeY[batchI];
        int sizeX = cumsizeX[batchI+1] - cumsizeX[batchI];
        if (idx >= sizeY) return;

        int nj = idx / n;
        int ni = idx - nj * n;
        int kj = nj / ksize;
        int ki = nj - kj * ksize;
        int xj = cumsizeX[batchI] - padding + ki + kj*stride;
        int xi = ni + xj * n;
        int yi = cumsizeY[batchI] + idx;
        if (xj < cumsizeX[batchI] || xj >= cumsizeX[batchI+1]) y[yi] = 0;
        else y[yi] = x[xi];
    }""")
    quote
        ydims = Array{Int}(undef, length(dims))
        for i = 1:length(dims)
            d = dims[i]
            k = (ksize - 1) * dilation + 1
            ydims[i] = (d + 2padding - k) ÷ stride + 1
        end
        y = similar(x, ksize*size(x,1), sum(ydims))
        cumsize_y = Array{Cint}(undef, length(dims)+1)
        cumsize_x = similar(cumsize_y)
        cumsize_y[1] = cumsize_x[1] = 0
        for i = 2:length(cumsize_y)
            cumsize_y[i] = cumsize_y[i-1] + size(y,1)*ydims[i-1]
            cumsize_x[i] = cumsize_x[i-1] + dims[i-1]
        end
        cumsize_y = CuArray(cumsize_y)
        cumsize_x = CuArray(cumsize_x)

        m = maximum(dims) * size(y,1)
        gdims, bdims = cudims(m*length(dims))
        $k(gdims, bdims, pointer(y), pointer(x), pointer(cumsize_y), pointer(cumsize_x),
            m, size(x,1), ksize, padding, stride)
        y
    end
end

function Base.cat(xs::CuArray{T}...; dims::Int) where T
    dim = dims
    length(xs) == 1 && return xs[1]
    N = max(dim, maximum(ndims,xs))
    cumdim = sum(x -> size(x,dim), xs)
    if ndims(xs[1]) == N
        ysize = Base.setindex(size(xs[1]), cumdim, dim)
    elseif ndims(xs[1])+1 == N
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
