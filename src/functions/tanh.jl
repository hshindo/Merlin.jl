import Base.tanh

"""
    tanh(x::Var)
"""
@graph function tanh(x::Var)
    y = tanh(x.data)
    df(gy) = isconst(x) || ∇tanh!(x.data, x.grad, y, gy)
    Var(y, [x], tanh, df)
end

tanh(x::CuArray) = CUDNN.activation!(CUDNN_ACTIVATION_TANH, x, similar(x))

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, gy, x, gx, beta=1.0)
end
