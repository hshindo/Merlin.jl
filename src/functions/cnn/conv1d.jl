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
    inchannel::Int
    outchannel::Int
    pad::Int
    stride::Int
    dilation::Int
    group::Int
end

function Conv1D{T}(::Type{T}, filtersize::Int, inchannel::Int, outchannel::Int, pad::Int, stride::Int; dilation=1, group=1)
    l = Linear(T, filtersize * inchannel, outchannel)
    Conv1D(l.w, l.b, filtersize, inchannel, outchannel, pad, stride, dilation, group)
end

function (f::Conv1D)(x::Var)
    h = window1d(x, f.filtersize, f.pad, f.stride, f.dilation)
    linear(f.w, h, f.b)
end

function (f::Conv1D)(x::Node)
    h = window1d(x, f.filtersize, f.pad, f.stride, f.dilation)
    Node(linear, f.w, h, f.b)
end
