export Linear, linear

type Linear
    w::Var
    b::Var
end

"""
    Linear(w::Var, b::Var)
    Linear(T::Type, indim::Int, outdim::Int)

Computes linear function (a.k.a. affine transformation).

* indim: size of inout dimension
* outdim: size of output dimension

```math
f(x) = W^{T}x + b
```
where ``W`` is a weight matrix and ``b`` is a bias vector.

```julia
T = Float32
x = Var(rand(T,10,5))
f = Linear(T,10,7)
y = f(x)
```
"""
function Linear{T}(::Type{T}, indim::Int, outdim::Int)
    r = sqrt(6 / (indim+outdim))
    w = uniform(T, -r, r, outdim, indim)
    b = fill(T(0), outdim, 1)
    Linear(zerograd(w), zerograd(b))
end

function (f::Linear)(x::Var)
    # setbackend
    linear(x, f.w, f.b)
end

function linear(x::Var, w::Var, b::Var)
    isa(x.data, Void) && return Var(nothing, linear, (x,w,b))
    y = w.data * x.data
    #y = similar(x.data, size(w.data,1), size(x.data,2))
    #fill!(y, 0)
    broadcast!(+, y, y, b.data)
    function df(gy)
        T = eltype(gy)
        isa(x.grad, Void) || BLAS.gemm!('T', 'N', T(1), w.data, gy, T(1), x.grad)
        isa(w.grad, Void) || BLAS.gemm!('N', 'T', T(1), gy, x.data, T(1), w.grad)
        isa(b.grad, Void) || add!(b.grad, sum(gy,2))
    end
    Var(y, df, (x,w,b))
end
