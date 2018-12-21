export AddAttention, AddAttention2

doc"""
    AddAttention(x1, x2)

```math
f(x_{i},x_{j}) = W^{T} \sigma (W_{1}x_{i} + W_{2}x_{j} + b) + b
```
"""
mutable struct AddAttention <: Functor
    l
end

function AddAttention(::Type{T}, insize::Int) where T
    l = Linear(T, insize, 3insize)
    AddAttention(l)
end

function (f::AddAttention)(x::Var, dims::Vector{Int})
    h = f.l(x)
    n = size(h,1) ÷ 3
    off = 0
    hs = Var[]
    for d in dims
        k = tanh(h[1:n,off+1:off+d])
        q = h[n+1:2n,off+1:off+d]
        kq = linear(k, q)
        a = softmax(kq)
        ha = h[2n+1:3n,off+1:off+d] * a
        push!(hs, ha)
        off += d
    end
    concat(2, hs...)
end

mutable struct AddAttention2 <: Functor
    l
    q
end

function AddAttention2(::Type{T}, insize::Int) where T
    l = Linear(T, insize, 2insize)
    q = parameter(Xavier()(T, insize, 1))
    AddAttention2(l, q)
end

function (f::AddAttention2)(x::Var, dims::Vector{Int})
    h = f.l(x)
    n = size(h,1) ÷ 2
    off = 0
    hs = Var[]
    for d in dims
        k = tanh(h[1:n,off+1:off+d])
        q = repeat(f.q, 1, d)
        # q = h[n+1:2n,off+1:off+d]
        #kq = linear(k, q)
        kq = linear(k, q)
        a = softmax(kq)
        ha = h[n+1:2n,off+1:off+d] * a
        push!(hs, ha)
        off += d
    end
    concat(2, hs...)
end
