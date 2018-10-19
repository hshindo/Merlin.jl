export Conv1d

doc"""
    Conv1d(T, ksize, inchannel, outchannel, [padding=0, stride=1, dilation=1])

1-dimensional convolution function.

```julia
T = Float32
x = Var(rand(T,10,5))
f = Conv1d(T, 5, 10, 3, padding=2)
y = f(x)
```
"""
mutable struct Conv1d
    ksize::Int
    padding::Int
    stride::Int
    dilation::Int
    W::Var
    b::Var
end

function Conv1d(::Type{T}, ksize::Int, inchannel::Int, outchannel::Int;
    padding=0, stride=1, dilation=1, init_W=Xavier(), init_b=Fill(0)) where T

    W = init_W(T, ksize*inchannel, outchannel)
    b = init_b(T, outchannel)
    Conv1d(ksize, padding, stride, dilation, parameter(W), parameter(b))
end

function (f::Conv1d)(x, dims)
    conv1d(x, dims, f.W, f.b, (ksize=f.ksize,padding=f.padding,stride=f.stride,dilation=f.dilation))
end

function conv1d(x::Var, dims, W::Var, b::Var, p)
    @assert ndims(x) == 2 && sum(dims) == size(x,2)
    h = conv1d_index(x, dims, p)
    y = linear(h, W, b)
    y
end
conv1d(x::Node, args...) = Node(conv1d, (x,args...))

function conv1d_index(x::Var, dims, p)
    ydata = conv1d_index(x.data, dims, p)
    Var(ydata, ∇conv1d_index!, (x,dims,p))
end

function conv1d_index(x::Matrix, dims::Vector{Int}, p::NamedTuple)
    ksize, padding, stride, dilation = p.ksize, p.padding, p.stride, p.dilation
    outdims = map(dims) do d
        k = (ksize - 1) * dilation + 1
        (d + 2padding - k) ÷ stride + 1
    end
    cumdim = 0
    h = Array{Int}(undef, ksize, sum(outdims))
    hi = 1
    for n = 1:length(dims)
        ndims = dims[n]
        i = cumdim - padding + 1
        for d = 1:outdims[n]
            for j = i:dilation:i+(ksize-1)*dilation
                h[hi] = cumdim < j <= cumdim+ndims ? j : 0
                hi += 1
            end
            i += stride
        end
        cumdim += ndims
    end
    lookup(x, h)
end

function ∇conv1d_index!(y::Var, x::Var, dims, p)
    ∇conv1d_index!(y.grad, x.grad, dims, p)
end

function ∇conv1d_index!(gy::Array, gx::Array, dims, p)
    ∇lookup!()
end

@generated function conv1d_index(x::CuMatrix{T}, dims::Vector{Int}, p::NamedTuple) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void conv1d_index($Ct *y, $Ct *x, int sizeY, int sizeX, int n, int ksize, int padding, int stride) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= sizeY) return;

        int vj = idx / n;
        int vi = idx - vj * n;
        int kj = vj / ksize;
        int ki = vj - kj * ksize;
        int xj = -padding + ki + kj * stride;
        int xi = vi + xj * n;
        if (xj < 0 || xj >= sizeX) y[idx] = 0;
        else y[idx] = x[xi];
    }""")
    quote
        ksize, padding, stride, dilation = p.ksize, p.padding, p.stride, p.dilation
        ydims = Array{Int}(undef, length(dims))
        for i = 1:length(dims)
            d = dims[i]
            k = (ksize - 1) * dilation + 1
            ydims[i] = (d + 2padding - k) ÷ stride + 1
        end
        y = similar(x, ksize*size(x,1), sum(ydims))
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(x), length(y), size(x,2), size(x,1), ksize, padding, stride)
        y
    end
end

@generated function ∇conv1d_index!(gy::CuMatrix{T}, gx::CuMatrix{T}, dims::Vector{Int}, p::NamedTuple) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void conv1d_index_grad($Ct *gy, $Ct *gx, int sizeY, int sizeX, int n, int ksize, int padding, int stride) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= sizeY) return;

        int vj = idx / n;
        int vi = idx - vj * n;
        int kj = vj / ksize;
        int ki = vj - kj * ksize;
        int xj = -padding + ki + kj * stride;
        int xi = vi + xj * n;
        if (xj >= 0 || xj < sizeX) atomicAdd(&gx[xi], gy[idx]);
    }""")
    quote
        ksize, padding, stride, dilation = p.ksize, p.padding, p.stride, p.dilation
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, pointer(gy), pointer(gx), length(gy), size(gx,2), size(gx,1), ksize, padding, stride)
    end
end
