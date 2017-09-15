export sigmoid

doc"""
    sigmoid(x)

Sigmoid logistic function.
```math
f(x) = (1 + \exp(-x))^{-1}
```
"""
sigmoid(x::Var) = Var(sigmoid(x.data), x.batchdims, sigmoid, (x,))

sigmoid(x::Node; name="sigmoid") = Node(sigmoid, x, name=name)

function sigmoid{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    y
end

function addgrad!(y::Var, ::typeof(sigmoid), x::Var)
    isvoid(x.grad) || ∇sigmoid!(y.data, y.grad, x.data, x.grad)
end

function ∇sigmoid!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end
