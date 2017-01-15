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

(f::Linear)(x::Var) = linear(x, f.w, f.b)

linear(x::Var, w::Var, b::Var) = forward(linear, x, w, b)

function forward{T}(::typeof(linear), x::Matrix{T}, w::Matrix{T}, b::Matrix{T})
    y = w * x
    broadcast!(+, y, y, b)
    function backward!{T}(gy::Matrix{T}, gx, gw, gb)
        isvoid(gx) || BLAS.gemm!('T', 'N', T(1), w, gy, T(1), gx)
        isvoid(gw) || BLAS.gemm!('N', 'T', T(1), gy, x, T(1), gw)
        isvoid(gb) || add!(gb, sum(gy,2))
    end
    y, backward!
end

export NormLinear
type NormLinear
    v::Var
    g::Var
    b::Var
end

function NormLinear{T}(::Type{T}, indim::Int, outdim::Int)
    row = outdim
    col = indim
    v = randn(T, row, col) * 0.05
    g = ones(T, col)
    b = fill(T(0), outdim, 1)
    NormLinear(zerograd(v), zerograd(g), zerograd(b))
end

(f::NormLinear)(x::Var) = normlinear(x, f.v, f.g, f.b)

function normlinear(x::Var, v::Var, g::Var, b::Var)
    w = normalize(v,g)
    linear(x, w, b)
end

function normweight{T}(::Type{T}, row::Int, col::Int)
    v = randn(T, row, col) * 0.05
    g = ones(T, row, 1)
    normalize(zerograd(v)) * zerograd(g)
end
