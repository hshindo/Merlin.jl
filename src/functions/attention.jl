export Attention

doc"""
    Attention(x1, x2)

```math
f(x_{i},x_{j}) = W^{T} \sigma (W_{1}x_{i} + W_{2}x_{j} + b) + b
```
"""
mutable struct Attention <: Functor
    linear
end

function Attention(::Type{T}, insize::Int) where T
    l = Linear(T, insize, 3insize)
    Attention(l)
end

function (f::Attention)(x::Var, dims::Vector{Int})
    h = f.linear_h(x)
    d = f.linear_d(x)
    repeat(d, 1, )

    a = tanh(linear(h,d,f.b))

end
