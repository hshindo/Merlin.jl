export Linear, linear

"""
    Linear(w::Var, x::Var, [b::Var])

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
type Linear
    w::Var
    b::Var
end

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
    x.data == nothing && return Var(nothing, (linear,w,x,b))
    T = eltype(x)
    y = Var(T, (size(w,1),size(x,2)), (w,x,b))
    BLAS.gemm!('N', 'N', T(1), w.data, x.data, T(0), y.data)
    broadcast!(.+, y.data, y.data, b.data)
    y.df = () -> isconst(x) || ∇linear!(y, w, x, b)
    y
end

function ∇linear!(y::Var, w::Var, x::Var, b::Var)
    T = eltype(y)
    isconst(w) || BLAS.gemm!('N', 'T', T(1), y.grad, x.data, T(1), w.grad)
    isconst(x) || BLAS.gemm!('T', 'N', T(1), w.data, y.grad, T(1), x.grad)
    if !isconst(b)
        for offset = 1:length(b.data):length(y.grad)
            BLAS.axpy!(length(b.data), T(1), pointer(y.grad,offset), 1, pointer(b.grad), 1)
        end
    end
end

function update!(f::Linear, opt)
    opt(f.w.data, f.w.grad)
    opt(f.b.data, f.b.grad)
end
