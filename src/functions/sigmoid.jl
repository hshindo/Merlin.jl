export sigmoid

"""
    sigmoid(x::Var)

Sigmoid function.
"""
function sigmoid(x::Var)
    x.data == nothing && return Var(nothing, sigmoid, (x,))
    y = sigmoid(x.data)
    df(gy) = isconst(x) || ∇sigmoid!(y, gy, x.data, x.grad)
    Var(y, sigmoid, (x,), df)
end

function sigmoid{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    y
end

sigmoid(x::CuArray) = CUDNN.activation(CUDNN_ACTIVATION_SIGMOID, x)

function ∇sigmoid!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

function ∇sigmoid!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_SIGMOID, y, gy, x, gx, beta=1.0)
end
