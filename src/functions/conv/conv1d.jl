export Conv1D

doc"""
    Conv1D(T, ksize, insize, outsize, pad, stride, [dilation=1, init_W=Xavier(), init_b=Fill(0)])

1-dimensional convolution function.

```julia
T = Float32
x = Var(rand(T,10,5))
f = Conv1D(T, 5, 10, 3, 2, 1)
y = f(x)
```
"""
mutable struct Conv1D
    W::Var
    b::Var
    ksize::Int
    pad::Int
    stride::Int
    dilation::Int
end

function Conv1D(::Type{T}, ksize::Int, insize::Int, outsize::Int, pad::Int, stride::Int;
    dilation=1, init_W=Xavier(), init_b=Fill(0)) where T

    W = init_W(T, ksize*insize, outsize)
    b = init_b(T, outsize)
    Conv1D(zerograd(W), zerograd(b), ksize, pad, stride, dilation)
end

(c::Conv1D)(x) = conv1d(x, c.W, c.b, c.ksize, c.pad, c.stride, c.dilation)

function conv1d(x::Var, W::Var, b::Var, ksize::Int, pad::Int, stride::Int, dilation::Int)
    batchdims = map(x.batchdims) do d
        (d + 2pad - ksize) ÷ stride + 1
    end
    h = zeros(eltype(x), size(x,1)*ksize, sum(batchdims))
    window1d!(h, x.data, batchdims, ksize, pad, stride, dilation)
    y = linear(h, W.data, b.data)
    Var(y, batchdims, conv1d, (x,W,b,h,ksize,pad,stride,dilation))
end

conv1d(x::Node, args...; name="") = Node(conv1d, (x,args...), name)

function addgrad!(y::Var, ::typeof(conv1d), x::Var, W::Var, b::Var, h, args...)
    gh = zeros(h)
    addgrad!(y, linear, Var(h,grad=gh), W, b)
    isvoid(x.grad) || ∇window1d!(gh, x.grad, x.batchdims, args...)
end

function window1d!(y::Matrix{T}, x::Matrix{T}, batchdims::Vector{Int}, ksize::Int, pad::Int, stride::Int, dilation::Int) where T
    yi = 1
    s = 1
    for dim in batchdims
        i = s - pad
        while i + ksize <= s + dim + pad
            for w = 0:ksize-1
                j = i + w * dilation
                if j >= s && j < s + dim
                    xi = (j-1) * size(x,1) + 1
                    for c = 0:size(x,1)-1
                        y[yi+c] = x[xi+c]
                    end
                end
                yi += size(x,1)
            end
            i += stride
        end
        s += dim
    end
end

function ∇window1d!(gy::Matrix{T}, gx::Matrix{T}, batchdims::Vector{Int}, ksize::Int, pad::Int, stride::Int, dilation::Int) where T
    yi = 1
    s = 1
    for dim in batchdims
        i = s - pad
        while i + ksize <= s + dim + pad
            for w = 0:ksize-1
                j = i + w * dilation
                if j >= s && j < s + dim
                    xi = (j-1) * size(gx,1) + 1
                    for c = 0:size(gx,1)-1
                        gx[xi+c] += gy[yi+c]
                    end
                end
                yi += size(gx,1)
            end
            i += stride
        end
        s += dim
    end
end
