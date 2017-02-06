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
    w = uniform(T, -r, r, indim, outdim)
    b = fill(T(0), outdim, 1)
    Linear(zerograd(w), zerograd(b))
end

(f::Linear)(x::Var) = linear(x, f.w, f.b)

linear(x::Var, w::Var, b::Var) = forward(linear, x, w, b)

function forward(::typeof(linear), x::Array, w::Array, b::Array)
    T = eltype(x)
    y = BLAS.gemm('T', 'N', T(1), w, x)
    broadcast!(.+, y, y, b)
    function backward!(gy, gx, gw, gb)
        isvoid(gx) || BLAS.gemm!('N', 'N', T(1), w, gy, T(1), gx)
        isvoid(gw) || BLAS.gemm!('N', 'T', T(1), x, gy, T(1), gw)
        isvoid(gb) || add!(gb, sum(gy,2))
    end
    y, backward!
end

export GatedLinear
function GatedLinear{T}(::Type{T}, indim::Int, outdim::Int)
    x = Var()
    #v1 = zerograd(randn(T,indim,outdim) * 0.05)
    #g1 = zerograd(ones(T,1,outdim))
    #w1 = normalize(v1) .* g1
    #b1 = zerograd(zeros(T,outdim))
    #y1 = linear(x, w1, b1)
    y1 = Linear(T,indim,outdim)(x)

    #v2 = zerograd(randn(T,indim,outdim) * 0.05)
    #g2 = zerograd(ones(T,1,outdim))
    #w2 = normalize(v1) .* g2
    #b2 = zerograd(zeros(T,outdim))
    #y2 = linear(x, w2, b2)
    y2 = Linear(T,indim,outdim)(x)

    h = y1 .* sigmoid(y2)
    Graph(h, x)
end

export NormLinear
type NormLinear
    v::Var
    g::Var
    b::Var
end

function NormLinear{T}(::Type{T}, indim::Int, outdim::Int)
    v = zerograd(randn(T,indim,outdim) * 0.05)
    g = zerograd(ones(T,1,outdim))
    b = zerograd(zeros(T,outdim))
    NormLinear(v, g, b)
end

(f::NormLinear)(x::Var) = normlinear(x, f.v, f.g, f.b)

function normlinear(x::Var, v::Var, g::Var, b::Var)
    w = normalize(v) .* g
    linear(x, w, b)
end
