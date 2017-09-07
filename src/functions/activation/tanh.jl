import Base.tanh

doc"""
    tanh(x)

Hyperbolic tangent function.
"""
tanh(x::Var) = Var(tanh.(x.data), tanh, (x,))

tanh(x::Node) = Node(tanh, x)

function addgrad!(y::Var, ::typeof(tanh), x::Var)
    isvoid(x.grad) && return
    ∇tanh!(y.data, y.grad, x.data, x.grad)
end

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
