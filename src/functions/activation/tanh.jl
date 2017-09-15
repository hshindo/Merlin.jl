import Base.tanh

doc"""
    tanh(x)

Hyperbolic tangent function.
"""
tanh(x::Var) = Var(tanh.(x.data), x.batchdims, tanh, (x,))

tanh(x::Node; name="tanh") = Node(tanh, x, name=name)

function addgrad!(y::Var, ::typeof(tanh), x::Var)
    isvoid(x.grad) || ∇tanh!(y.data, y.grad, x.data, x.grad)
end

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
