export Linear, linear

type Linear
    w::Var
    b::Var
end

"""
    Linear(w::Var, b::Var)
    Linear(T::Type, indim::Int, outdim::Int)

Compute linear function (a.k.a. affine transformation).

```math
f(x) = W^{T}x + b
```
where ``W`` is a weight matrix and ``b`` is a bias vector.

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = Linear(Float32,10,7)
y = f(x)
```
"""
function Linear(T::Type, indim::Int, outdim::Int)
    r = T(sqrt(6 / (indim+outdim)))
    w = rand(T, outdim, indim)
    w .*= 2r
    w .-= r
    b = fill(T(0), outdim, 1)
    Linear(zerograd(w), zerograd(b))
end

function (f::Linear)(x::Var)
    w, b = f.w, f.b
    y = w.data * x.data
    broadcast!(+, y, y, b.data)
    df(gy) = ∇linear!(y, gy, w, x, b)
    Var(y, df, (w,x,b))
end
(f::Linear)(x::Var{Void}) = Var(Void(), f, (x,))

function ∇linear!(y, gy, w::Var, x::Var, b::Var)
    T = eltype(y)
    w.grad == nothing || BLAS.gemm!('N', 'T', T(1), gy, x.data, T(1), w.grad)
    x.grad == nothing || BLAS.gemm!('T', 'N', T(1), w.data, gy, T(1), x.grad)
    broadcast!(+, b.grad, b.grad, sum(gy,2))
end
