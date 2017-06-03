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
    Linear(param(w), param(b))
end

(f::Linear)(x::Var) = Var(nothing, (f,x))

function forward!(y::Var, f::Linear, x::Var)
    y.data = f.w.data * x.data
    y.df! = () -> begin
        T = eltype(f.w.data)
        BLAS.gemm!('N', 'T', T(1), y.grad, x.data, T(1), f.w.grad)
        isvoid(x.grad) || BLAS.gemm!('T', 'N', T(1), f.w.data, y.grad, T(1), x.grad)
    end
end

#(f::Linear)(x::Var) = f.w * x .+ f.b

#=
(f::Linear)(x::Var) = linear(x, f.w, f.b)

linear(x::Var, w::Var, b::Var) = w * x .+ b
=#



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
