export Linear
export linear

struct Linear <: Functor
    W::Var
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
    init_W=Xavier(), init_b=Fill(0)) where T

    W = init_W(T, insize, outsize)
    b = init_b(T, outsize)
    Linear(parameter(W), parameter(b))
end

(f::Linear)(x) = linear(x, f.W, f.b)

function linear(x::Var, W::Var, b=nothing)
    T = eltype(x)
    if ndims(x) == 1
        ydata = gemv('T', W.data, x.data)
        b == nothing || addto!(ydata, b.data)
    elseif ndims(x) == 2
        ydata = gemm('T', 'N', W.data, x.data)
        b == nothing || broadcast_addto!(ydata, b.data)
    else
        throw("Invalid ndims of x: $(ndims(x))")
    end
    Var(ydata, ∇linear!, (x,W,b))
end

function ∇linear!(y::Var, x::Var, W::Var, b)
    T = eltype(x)
    if ndims(x) == 1
        gy = reshape(y.grad, length(y.grad), 1)
        xdata = reshape(x.data, length(x.data), 1)
        isnothing(W.grad) || gemm!('N', 'T', T(1), xdata, gy, T(1), W.grad)
        isnothing(x.grad) || gemv!('N', T(1), W.data, y.grad, T(1), x.grad)
        isnothing(b) || isnothing(b.grad) || addto!(b.grad, y.grad)
    elseif ndims(x) == 2
        isnothing(x.grad) || gemm!('N', 'N', T(1), W.data, y.grad, T(1), x.grad)
        isnothing(W.grad) || gemm!('N', 'T', T(1), x.data, y.grad, T(1), W.grad)
        isnothing(b) || isnothing(b.grad) || addto!(b.grad, sum(y.grad,dims=2))
    else
        throw("Invalid ndims of x: $(ndims(x))")
    end
end
