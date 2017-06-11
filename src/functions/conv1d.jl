export Conv1D

"""
    Conv1D(T::Type, insize::Int, outsize::Int, pad::Int, stride::Int)

1-dimensional convolution function.

## 👉 Example
```julia
f = Conv1D(x, 10, 0, 1)
x = Var(rand(Float32,10,5))
y = f(x)
```
"""
type Conv1D
    w::Var
    b::Var
    insize::Int
    outsize::Int
    pad::Int
    stride::Int
end

function Conv1D{T}(::Type{T}, insize::Int, outsize::Int, pad::Int, stride::Int)
    w = uniform(T, -0.001, 0.001, outsize, insize)
    b = zeros(T, outsize, 1)
    Conv1D(zerograd(w), zerograd(b), insize, outsize, pad, stride)
end

function (f::Conv1D)(x::Var)
    h = window1d(x, f.insize, f.pad, f.stride)
    y = f.w * h .+ f.b
    y
end
