export l2

doc"""
    l2(x::Var, lambda::Float64)

L2 regularization.

```math
y = \frac{\lambda}{2}\left\Vert \mathbf{x} \right\Vert ^{2}
```

```julia
x = Var(rand(Float32,10,5))
y = l2(x, 0.01)
```
"""
function l2(x::Var, lambda::Float64)
    isnothing(x.data) && return Var(nothing,l2,(x,lambda))
    T = eltype(x)
    y = mapreduce(x -> x*x, +, x.data) * T(lambda) / 2
    Var([y], (l2,x,lambda))
end

function addgrad!(y::Var, ::typeof(l2), x::Var, lambda::Float64)
    T = eltype(y)
    isvoid(y.grad) || ∇l2!(y.grad, x.data, x.grad, T(lambda))
end

function ∇l2!(gy::Vector{T}, x::Array{T}, gx::Array{T}, lambda::T) where T
    @inbounds for i = 1:length(x)
        gx[i] += gy[1] * lambda * x[i]
    end
end
