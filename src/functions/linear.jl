export Linear
export linear

struct Linear
    w::Var
    b::Var
end

"""
    Linear(T::Type, insize::Int, outsize::Int, [init_w=Xavier()], [init_b=Zeros()])

Linear function (a.k.a. affine transformation).

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
function Linear{T}(::Type{T}, insize::Int, outsize::Int; init_w=Xavier(), init_b=Zeros())
    w = init_w(T, insize, outsize)
    b = init_b(T, outsize)
    Linear(Var(w,hasgrad=true), Var(b,hasgrad=true))
end

(f::Linear)(x) = linear(x, f.w, f.b)

function linear(x::Var, w::Var, b::Var)
    y = linear(x.data, w.data, b.data)
    Var(y, x.batchdims, linear, (x,w,b))
end

linear(x::Node, w, b; name="linear") = Node(linear, x, w, b, name=name)

function linear(x::Matrix, w::Matrix, b::Vector)
    y = gemm('T', 'N', w, x)
    broadcast!(+, y, y, b)
end

function addgrad!(y::Var, ::typeof(linear), x::Var, w::Var, b::Var)
    ∇linear!(y.data, y.grad, x.data, x.grad, w.data, w.grad, b.data, b.grad)
end

function ∇linear!{T}(y::Matrix{T}, gy::Matrix, x::Matrix, gx::Matrix, w, gw, b, gb)
    isvoid(gx) || BLAS.gemm!('N', 'N', T(1), w, gy, T(1), gx)
    isvoid(gw) || BLAS.gemm!('N', 'T', T(1), x, gy, T(1), gw)
    if !isvoid(gb)
        g = sum(gy, 2)
        BLAS.axpy!(T(1), g, gb)
    end
end

export NormLinear
struct NormLinear
    w
    b
end

function NormLinear{T}(::Type{T}, insize::Int, outsize::Int)
    w = randn(T,outsize,insize) * T(1 / insize)
    b = fill(T(0), outsize)
    Linear(zerograd(w), zerograd(b))
end

(f::NormLinear)(x) = normlinear(f.w, x, f.b)

function normlinear(w::Var, x::Var, b::Var)
    y = w.data * x.data .+ b.data
    Var(y, normlinear, (w,x,b))
end

function addgrad!(y::Var, ::typeof(normlinear), w::Var, x::Var, b::Var)
    T = eltype(y.data)
    isvoid(w.grad) || BLAS.gemm!('N', 'T', T(1), y.grad, x.data, T(1), w.grad)
    isvoid(x.grad) || BLAS.gemm!('T', 'N', T(1), w.data, y.grad, T(1), x.grad)
    if !isvoid(b.grad)
        g = sum(y.grad, 2)
        BLAS.axpy!(T(1), g, b.grad)
    end
end

#=
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
=#
