doc"""
    tanh(x::Var)

Hyperbolic tangent function.
"""
function Base.tanh(x::Var)
    configure!(x)
    Var(tanh(x.data), (tanh,x))
end
Base.tanh(x::Array) = tanh.(x)
Base.tanh(x::CuArray) = CUDNN.tanh(x)

function addgrad!(y::Var, ::typeof(tanh), x::Var)
    isvoid(x.grad) && return
    ∇tanh!(y.data, y.grad, x.data, x.grad)
end

function ∇tanh!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end

∇tanh!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray) = CUDNN.∇tanh!(y, gy, x, gx)
