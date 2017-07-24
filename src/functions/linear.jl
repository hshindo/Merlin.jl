export Linear

type Linear
    w::Var
    b::Var
end

"""
    Linear(T::Type, insize::Int, outsize::Int)

Computes linear function (a.k.a. affine transformation).

* insize: size of input dimension
* outsize: size of output dimension

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
function Linear{T}(::Type{T}, insize::Int, outsize::Int)
    r = sqrt(6 / (insize+outsize))
    w = uniform(T, -r, r, outsize, insize)
    b = fill(T(0), outsize, 1)
    Linear(zerograd(w), zerograd(b))
end

(f::Linear)(x::Var) = f.w * x .+ f.b


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
