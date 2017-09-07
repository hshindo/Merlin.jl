export Conv1D

doc"""
    Conv1D(T, ksize, insize, outsize, pad, stride; dilation=1)

1-dimensional convolution function.

# 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = Conv1D(x, 5, 10, 3, 0, 1)
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

function Conv1D{T}(::Type{T}, ksize, insize, outsize, pad, stride; dilation=1)
    w = randn(T,outsize,ksize*insize) * T(sqrt(2 / (ksize+insize+outsize)))
    b = fill(T(0), outsize)
    Conv1D(zerograd(w), zerograd(b), ksize, pad, stride, dilation)
end
(c::Conv1D)(x) = conv1d(x, c.w, c.b, c.ksize, c.pad, c.stride, c.dilation)
(c::Conv1D)(x, batchdims) = conv1d_batch(x, batchdims, c.w, c.b, c.ksize, c.pad, c.stride, c.dilation)

function conv1d(x, w, b, ksize, pad, stride, dilation)
    h = window1d(x, ksize, pad, stride, dilation)
    linear(w, h, b)
end

function conv1d_batch(x, batchdims, w, b, ksize, pad, stride, dilation)
    h = window1d_batch(x, batchdims, ksize, pad, stride, dilation)
    linear(w, h, b)
end
