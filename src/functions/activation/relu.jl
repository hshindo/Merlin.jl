export relu

doc"""
    relu(x::Var)

Rectified Linear Unit.

```math
f(x) = \max(0, x)
```
"""
relu(x::Var) = Var(relu(x.data), ∇relu!, (x,))
relu(x::T) where T<:AbstractFloat = max(x, zero(T))
relu(x::Array) = relu.(x)
relu(x::CuArray) = CUDNN.relu(x)
relu(x::Node) = Node(relu, (x,))

function ∇relu!(y::Var, x::Var)
    isnothing(x.grad) && return
    ∇relu!(y.data, y.grad, x.data, x.grad)
end

function ∇relu!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(x)
        gx[i] += x[i] > T(0) ? gy[i] : T(0)
    end
end

∇relu!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray) = CUDNN.∇relu!(y, gy, x, gx)
