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
(f::Linear)(x::Var) = linear(f.w, x, f.b)

function linear(w::Var, x::Var, b::Var)
    x.data == nothing && return Var(nothing, linear, (w,x,b))
    y = w.data * x.data
    y .+= b.data
    df(gy) = ∇linear!(y, gy, w, x, b)
    Var(y, linear, (w,x,b), df)
end

function ∇linear!(y::Array, gy::Array, w::Var, x::Var, b::Var)
    T = eltype(y)
    isconst(w) || BLAS.gemm!('N', 'T', T(1), gy, x.data, T(1), w.grad)
    isconst(x) || BLAS.gemm!('T', 'N', T(1), w.data, gy, T(1), x.grad)
    if !isconst(b)
        for offset = 1:length(b.data):length(gy)
            BLAS.axpy!(length(b.data), T(1), pointer(gy,offset), 1, pointer(b.grad), 1)
        end
    end
end
