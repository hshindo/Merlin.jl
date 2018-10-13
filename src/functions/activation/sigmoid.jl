export sigmoid

doc"""
    sigmoid(x)

Sigmoid logistic function.

```math
f(x) = (1 + \exp(-x))^{-1}
```
"""
sigmoid(x::Var) = Var(sigmoid(x.data), ∇sigmoid!, (x,))
sigmoid(x::Array) = sigmoid.(x)
sigmoid(x::T) where T<:AbstractFloat = T(1 / (1 + exp(-x)))
sigmoid(x::CuArray) = CUDNN.sigmoid(x)
sigmoid(x::Node) = Node(sigmoid, (x,))

function ∇sigmoid!(y::Var, x::Var)
    isnothing(x.grad) && return
    ∇sigmoid!(y.data, y.grad, x.data, x.grad)
end

function ∇sigmoid!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

∇sigmoid!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray) = CUDNN.∇sigmoid!(y, gy, x, gx)
