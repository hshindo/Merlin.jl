export window1d

function window1d(x::Var, dims, ksize::Int; padding=0, stride=1, dilation=1)
    @assert padding >= 0 && stride > 0 && dilation > 0
    x = pack(x, dims, 0)
    ydata = window1d(x.data, ksize, padding, stride, dilation)
    y = Var(ydata, ∇window1d!, (x,ksize,padding,stride,dilation))

    ydims = map(dims) do d
        k = (ksize - 1) * dilation + 1
        ydim = (d + 2padding - k) ÷ stride + 1
        max(ydim, 0)
    end
    unpack(y, ydims)
end

function window1d(x::Array{T,3}, ksize::Int, padding::Int, stride::Int, dilation::Int) where T
    outdims = map(dims) do d
        k = (ksize - 1) * dilation + 1
        (d + 2padding - k) ÷ stride + 1
    end

    k = (ksize - 1) * dilation + 1
    outdim = (size(x,2) + 2padding - k) ÷ stride + 1

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

@generated function window1d(x::CuArray{T,3}, ksize::Int, padding::Int, stride::Int, dilation::Int) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void window1d($Ct *y, $Ct *x,
        int xsize1, int xsize2, int ysize1, int ysize2, int batchsize,
        int ksize, int padding, int stride, int dilation) {

        int yi = blockIdx.x * blockDim.x + threadIdx.x;
        int batchIdx = yi / (ysize1*ysize2);
        if (batchIdx >= batchsize) return;
        int idx = yi - batchIdx*ysize1*ysize2;

        int n2 = idx / xsize1;
        int n1 = idx - n2 * xsize1;
        int k2 = n2 / ksize;
        int k1 = n2 - k2 * ksize;

        int x2 = -padding + k1*dilation + k2*stride;
        int xi = n1 + x2*xsize1 + batchIdx*xsize1*xsize2;
        if (x2 >= 0 && x2 < xsize2) y[yi] = x[xi];
        else y[yi] = 0;
    }""")
    quote
        k = (ksize - 1) * dilation + 1
        ydim = (size(x,2) + 2padding - k) ÷ stride + 1
        y = similar(x, ksize*size(x,1), ydim, size(x,3))

        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(x),
            size(x,1), size(x,2), size(y,1), size(y,2), size(x,3),
            ksize, padding, stride, dilation)
        y
    end
end

function ∇window1d!(y::Var, x::Var, args...)
    isnothing(x.grad) || ∇window1d!(y.grad, x.grad, args...)
end

function ∇window1d!(gy::Matrix{T}, gx::Matrix{T}, dims::Vector{Int}, ksize::Int, padding::Int, stride::Int, dilation::Int) where T
    throw("Not implemented yet.")
end

@generated function ∇window1d!(gy::CuArray{T,3}, gx::CuArray{T,3}, ksize::Int, padding::Int, stride::Int, dilation::Int) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void window1d_grad($Ct *gy, $Ct *gx,
        int xsize1, int xsize2, int ysize1, int ysize2, int batchsize,
        int ksize, int padding, int stride, int dilation) {

        int yi = blockIdx.x * blockDim.x + threadIdx.x;
        int batchIdx = yi / (ysize1*ysize2);
        if (batchIdx >= batchsize) return;
        int idx = yi - batchIdx*ysize1*ysize2;

        int n2 = idx / xsize1;
        int n1 = idx - n2 * xsize1;
        int k2 = n2 / ksize;
        int k1 = n2 - k2 * ksize;

        int x2 = -padding + k1*dilation + k2*stride;
        int xi = n1 + x2*xsize1 + batchIdx*xsize1*xsize2;
        if (x2 >= 0 && x2 < xsize2) atomicAdd(&gx[xi], gy[yi]);
    }""")
    quote
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, pointer(gy), pointer(gx),
            size(gx,1), size(gx,2), size(gy,1), size(gy,2), size(gx,3),
            ksize, padding, stride, dilation)
    end
end
