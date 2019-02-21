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
mutable struct Conv1d <: Functor
    ksize::Int
    padding::Int
    stride::Int
    dilation::Int
    ngroups::Int
    W::Var
    b::Var
end

function Conv1d(::Type{T}, ksize::Int, inchannel::Int, outchannel::Int;
    padding=0, stride=1, dilation=1, ngroups=1, init_W=Xavier(), init_b=Fill(0)) where T

    n = inchannel ÷　ngroups
    @assert ngroups * n == inchannel
    W = init_W(T, ksize*n, outchannel)
    b = init_b(T, outchannel)
    Conv1d(ksize, padding, stride, dilation, ngroups, parameter(W), parameter(b))
end

function (f::Conv1d)(x::Var, dims)
    n = size(f.W,1) ÷ f.ksize
    if f.ngroups == 1
        h = window1d(x, dims, f.ksize, padding=f.padding, stride=f.stride, dilation=f.dilation)
    else
        hs = Var[]
        for i = 1:f.ngroups
            g = x[(i-1)*n+1:i*n, :]
            h = window1d(g, dims, f.ksize, padding=f.padding, stride=f.stride, dilation=f.dilation)
            push!(hs, h)
        end
        h = concat(1, hs...)
    end
    W = concat(1, [f.W for i=1:f.ngroups]...)
    linear(h, W, f.b)
end
