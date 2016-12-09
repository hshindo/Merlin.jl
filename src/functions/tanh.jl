import Base.tanh

"""
    tanh(x::Var)

Hyperbolic tangent function.
"""
function tanh(x::Var)
    x.data == nothing && return Var(nothing, tanh, (x,))
    y = tanh(x.data)
    df(gy) = isconst(x) || ∇tanh!(y, gy, x.data, x.grad)
    Var(y, tanh, (x,), df)
end

tanh(x::CuArray) = CUDNN.activation(CUDNN_ACTIVATION_TANH, x)

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end

function ∇tanh!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, gy, x, gx, beta=1.0)
end
