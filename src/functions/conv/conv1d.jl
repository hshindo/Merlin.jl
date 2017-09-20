export Conv1D

doc"""
    Conv1D(T, ksize, insize, outsize, pad, stride; dilation=1, [init_w=Xavier()], [init_b=Zeros()])

1-dimensional convolution function.

```julia
x = Var(rand(Float32,10,5))
f = Conv1D(Float32, 5, 10, 3, 2, 1)
y = f(x)
```
"""
mutable struct Conv1D
    w::Var
    b::Var
    ksize::Int
    pad::Int
    stride::Int
    dilation::Int
end

function Conv1D{T}(::Type{T}, ksize::Int, insize::Int, outsize::Int, pad::Int, stride::Int;
    dilation=1, init_w=Xavier(), init_b=Zeros())

    w = init_w(T, ksize*insize, outsize)
    b = init_b(T, outsize)
    Conv1D(Var(w,hasgrad=true), Var(b,hasgrad=true), ksize, pad, stride, dilation)
end

(c::Conv1D)(x) = conv1d(x, c.w, c.b, c.ksize, c.pad, c.stride, c.dilation)

function conv1d(x::Var, w::Var, b::Var, ksize::Int, pad::Int, stride::Int, dilation::Int)
    batchdims = map(x.batchdims) do d
        (d + 2pad - ksize) ÷ stride + 1
    end
    h = zeros(eltype(x.data), size(x,1)*ksize, sum(batchdims))
    window1d!(h, x.data, batchdims, ksize, pad, stride, dilation)
    y = linear(h, w.data, b.data)
    Var(y, batchdims, conv1d, (x,w,b,h,ksize,pad,stride,dilation))
end

conv1d(x::Node, args...; name="conv1d") = Node(conv1d, x, args..., name=name)

function addgrad!(y::Var, ::typeof(conv1d), x::Var, w::Var, b::Var, h, args...)
    gh = zeros(h)
    ∇linear!(y.data, y.grad, h, gh, w.data, w.grad, b.data, b.grad)
    isvoid(x.grad) || ∇window1d!(gh, x.grad, x.batchdims, args...)
end

function window1d!{T}(y::Matrix{T}, x::Matrix{T}, batchdims::Vector{Int}, ksize::Int, pad::Int, stride::Int, dilation::Int)
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

function ∇window1d!{T}(gy::Matrix{T}, gx::Matrix{T}, batchdims::Vector{Int}, ksize::Int, pad::Int, stride::Int, dilation::Int)
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
