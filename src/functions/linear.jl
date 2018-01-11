export Linear
export linear

struct Linear <: Functor
    w::Var
    b::Var
end

"""
    Linear(T::Type, insize::Int, outsize::Int, [init_w=Xavier()], [init_b=Fill(0)])

Linear function (a.k.a. affine transformation).

```math
f(x) = w^{T}x + b
```
where ``w`` is a weight matrix and ``b`` is a bias vector.

```julia
T = Float32
x = Var(rand(T,10,5))
f = Linear(T,10,7)
y = f(x)
```
"""
function Linear(::Type{T}, insize::Int, outsize::Int;
    init_w=Xavier(), init_b=Fill(0)) where T

    w = init_w(T, insize, outsize)
    b = init_b(T, outsize)
    Linear(zerograd(w), zerograd(b))
end
(f::Linear)(x) = linear(x, f.w, f.b)

function linear(x::Var, w::Var, b::Var)
    y = BLAS.gemm('T', 'N', w.data, x.data)
    y .+= b.data
    Var(y, (linear,x,w,b))
end
linear(x::Node, w, b) = Node(linear, x, w, b)

function addgrad!(y::Var, ::typeof(linear), x::Var, w::Var, b::Var)
    T = eltype(y)
    isvoid(x.grad) || BLAS.gemm!('N', 'N', T(1), w.data, y.grad, T(1), x.grad)
    isvoid(w.grad) || BLAS.gemm!('N', 'T', T(1), x.data, y.grad, T(1), w.grad)
    isvoid(b.grad) || BLAS.axpy!(T(1), sum(y.grad,2), b.grad)
end

function Base.convert(backend, linear::Linear)
    w = convert(backend, linear.w)
    b = convert(backend, linear.b)
    Linear(w, b)
end

getparams(f::Linear) = f.w, f.b
