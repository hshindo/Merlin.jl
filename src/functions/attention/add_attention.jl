export AddAttention

doc"""
    AddAttention(x1, x2)

```math
f(x_{i},x_{j}) = W^{T} \sigma (W_{1}x_{i} + W_{2}x_{j} + b) + b
```
"""
struct AddAttention
    l1
    l2
end

function AddAttention(::Type{T}, insize::Int, outsize::Int) where T
    l1 = Linear(T, 2insize, insize)
    l2 = Linear(T, insize, outsize)
    AddAttention(l1, l2)
end

function (f::AddAttention)(x1::Var, x2::Var)
    h = pairwise(x1, x2)
    h = f.l1(h)
    h = relu(h)
    f.l2(h)
end
