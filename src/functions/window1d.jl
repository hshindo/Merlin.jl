export window1d

function window1d(x::Node, dims, ksize, padding, stride, dilation)
    Node(window1d, (x,dims,ksize,padding,stride,dilation))
end

function window1d(x::Var, dims, ksize, padding, stride, dilation)
    ydata = window1d(x.data, dims, ksize, padding, stride, dilation)
    Var(ydata, ∇window1d!, (x,dims,ksize,padding,stride,dilation))
end

function window1d(x::Matrix{T}, dims::Vector{Int}, ksize::Int, padding::Int, stride::Int, dilation::Int) where T
    outdims = map(dims) do d
        k = (ksize - 1) * dilation + 1
        (d + 2padding - k) ÷ stride + 1
    end
    cumdim = 0
    y = similar(x, ksize*size(x,1), sum(outdims))
    fill!(y, 0)
    yi = 1
    for n = 1:length(dims)
        ndims = dims[n]
        i = cumdim - padding + 1
        for d = 1:outdims[n]
            for j = i:dilation:i+(ksize-1)*dilation
                if cumdim < j <= cumdim+ndims
                    copyto!(y, yi, x, (j-1)*size(x,1)+1, size(x,1))
                end
                yi += size(x, 1)
            end
            i += stride
        end
        cumdim += ndims
    end
    y
end

@generated function window1d(x::CuMatrix{T}, dims::Vector{Int}, ksize, padding, stride, dilation) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void window1d($Ct *y, $Ct *x, int *cumsizeY, int *cumsizeX,
        int batchsize, int m, int n, int ksize, int padding, int stride) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batchI = idx / m;
        if (batchI >= batchsize) return;
        idx -= m * batchI;
        int sizeY = cumsizeY[batchI+1] - cumsizeY[batchI];
        if (idx >= sizeY) return;

        int nj = idx / n;
        int ni = idx - nj * n;
        int kj = nj / ksize;
        int ki = nj - kj * ksize;
        int xj = cumsizeX[batchI] - padding + ki + kj*stride;
        int xi = ni + xj * n;
        int yi = idx + cumsizeY[batchI];
        if (xj >= cumsizeX[batchI] && xj < cumsizeX[batchI+1]) y[yi] = x[xi];
        else y[yi] = 0;
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

        m = maximum(ydims) * size(y,1)
        gdims, bdims = cudims(m*length(ydims))
        $k(gdims, bdims, pointer(y), pointer(x), pointer(cumsize_y), pointer(cumsize_x),
            length(dims), m, size(x,1), ksize, padding, stride)
        y
    end
end

function ∇window1d!(y::Var, x::Var, dims, args...)
    isnothing(x.grad) || ∇window1d!(y.grad, x.grad, dims, args...)
end

function ∇window1d!(gy::Matrix{T}, gx::Matrix{T}, dims::Vector{Int}, ksize::Int, padding::Int, stride::Int, dilation::Int) where T
    outdims = map(dims) do d
        k = (ksize - 1) * dilation + 1
        (d + 2padding - k) ÷ stride + 1
    end
    cumdim = 0
    yi = 1
    for n = 1:length(dims)
        ndims = dims[n]
        i = cumdim - padding + 1
        for d = 1:outdims[n]
            for j = i:dilation:i+(ksize-1)*dilation
                if cumdim < j <= cumdim+ndims
                    addto!(gx, (j-1)*size(gx,1)+1, gy, yi, size(gx,1))
                end
                yi += size(gx, 1)
            end
            i += stride
        end
        cumdim += ndims
    end
end

@generated function ∇window1d!(gy::CuMatrix{T}, gx::CuMatrix{T}, dims::Vector{Int}, ksize, padding, stride, dilation) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void window1d_grad($Ct *gy, $Ct *gx, int *cumsizeY, int *cumsizeX,
        int batchsize, int m, int n, int ksize, int padding, int stride) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int batchI = idx / m;
        if (batchI >= batchsize) return;
        idx -= m * batchI;
        int sizeY = cumsizeY[batchI+1] - cumsizeY[batchI];
        if (idx >= sizeY) return;

        int nj = idx / n;
        int ni = idx - nj * n;
        int kj = nj / ksize;
        int ki = nj - kj * ksize;
        int xj = cumsizeX[batchI] - padding + ki + kj*stride;
        int xi = ni + xj * n;
        int yi = idx + cumsizeY[batchI];
        if (xj >= cumsizeX[batchI] && xj < cumsizeX[batchI+1]) atomicAdd(&gx[xi], gy[yi]);
    }""")
    quote
        ydims = Array{Int}(undef, length(dims))
        for i = 1:length(dims)
            d = dims[i]
            k = (ksize - 1) * dilation + 1
            ydims[i] = (d + 2padding - k) ÷ stride + 1
        end
        cumsize_y = Array{Cint}(undef, length(dims)+1)
        cumsize_x = similar(cumsize_y)
        cumsize_y[1] = cumsize_x[1] = 0
        for i = 2:length(cumsize_y)
            cumsize_y[i] = cumsize_y[i-1] + size(gy,1)*ydims[i-1]
            cumsize_x[i] = cumsize_x[i-1] + dims[i-1]
        end
        cumsize_y = CuArray(cumsize_y)
        cumsize_x = CuArray(cumsize_x)

        m = maximum(ydims) * size(gy,1)
        gdims, bdims = cudims(m*length(ydims))
        $k(gdims, bdims, pointer(gy), pointer(gx), pointer(cumsize_y), pointer(cumsize_x),
            length(dims), m, size(gx,1), ksize, padding, stride)
    end
end
