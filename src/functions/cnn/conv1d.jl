export Conv1D

"""
    Conv1D(T::Type, insize::Int, outsize::Int, pad::Int, stride::Int; dilation::Int)

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
    filtersize::Int
    outsize::Int
    pad::Int
    stride::Int
    dilation::Int
end

function Conv1D(::Type{T}, filtersize::Int, outsize::Int, pad::Int, stride::Int; dilation=1) where {T}
    l = Linear(T, filtersize, outsize)
    Conv1D(l.w, l.b, filtersize, outsize, pad, stride, dilation)
end

function (f::Conv1D)(x::Var)
    h = window1d(x, f.filtersize, f.pad, f.stride, f.dilation)
    linear(f.w, h, f.b)
end
(f::Conv1D)(x::Node) = Node(f, x)
