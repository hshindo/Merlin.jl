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
    h = window1d(x, dims, f.ksize, f.padding, f.stride, f.dilation)
    linear(h, f.W, f.b)
end
