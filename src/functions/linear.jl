export Linear
export linear

struct Linear
    W::Var
    b::Var
end

"""
    Linear(T::Type, insize::Int, outsize::Int, [init_W=Xavier()], [init_b=Fill(0)])

Linear function (a.k.a. affine transformation).

```math
f(x) = W^{T}x + b
```
where ``W`` is a weight matrix and ``b`` is a bias vector.

# Example
```julia
T = Float32
x = Var(rand(T,10,5))
f = Linear(T,10,7)
y = f(x)
```
"""
function Linear(::Type{T}, insize::Int, outsize::Int; init_W=Xavier(), init_b=Fill(0)) where T
    W = init_W(T, insize, outsize)
    b = init_b(T, outsize)
    Linear(zerograd(W), zerograd(b))
end
(f::Linear)(x) = linear(x, f.W, f.b)

function linear(x::Var, W::Var, b::Var)
    y = linear(x.data, W.data, b.data)
    Var(y, linear, (x,W,b))
end
linear(x::Node, W::Var, b::Var; name="") = Node(linear, (x,W,b), name)

function linear(x::Matrix, W::Matrix, b)
    y = BLAS.gemm('T', 'N', W, x)
    b == nothing || broadcast!(+, y, y, b)
    y
end

function addgrad!(y::Var, ::typeof(linear), x::Var, W::Var, b::Var)
    T = eltype(y)
    isvoid(x.grad) || BLAS.gemm!('N', 'N', T(1), W.data, y.grad, T(1), x.grad)
    isvoid(W.grad) || BLAS.gemm!('N', 'T', T(1), x.data, y.grad, T(1), W.grad)
    isvoid(b.grad) || BLAS.axpy!(T(1), sum(y.grad,2), b.grad)
end
