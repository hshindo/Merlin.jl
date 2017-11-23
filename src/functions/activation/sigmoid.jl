export sigmoid

doc"""
    sigmoid(x)

Sigmoid logistic function.
```math
f(x) = (1 + \exp(-x))^{-1}
```
"""
sigmoid(x::Var) = Var(sigmoid.(x.data), x.batchdims, sigmoid, (x,))

sigmoid(x::Node; name="") = Node(sigmoid, (x,), name)

sigmoid(x::T) where T<:AbstractFloat = 1 / (1 + exp(-x))

function addgrad!(y::Var, ::typeof(sigmoid), x::Var)
    isvoid(x.grad) || ∇sigmoid!(y.data, y.grad, x.data, x.grad)
end

# ∇sigmoid!(y::T, gy::T, x::T, gx::T) where T = gy * y[i] * (T(1) - y[i])

function ∇sigmoid!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end
