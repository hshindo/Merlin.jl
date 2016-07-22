export relu, sigmoid
import Base.tanh

"""
    relu(x)
"""
function relu(x::Var)
    y = relu(x.data)
    df(gy) = hasgrad(x) && ∇relu!(x.data, x.grad, y, gy)
    Var(y, [x], df)
end

function relu{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    y
end

relu(x::CuArray) = activation!(CUDNN_ACTIVATION_RELU, x, similar(x))

function ∇relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end

function ∇relu!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    ∇activation!(CUDNN_ACTIVATION_RELU, y, gy, x, gx, beta=1.0)
end

"""
    sigmoid(x)
"""
function sigmoid(x::Var)
    y = sigmoid(x.data)
    df(gy) = hasgrad(x) && ∇sigmoid!(x.data, x.grad, y, gy)
    Var(y, [x], df)
end

function sigmoid{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    y
end

sigmoid(x::CuArray) = activation!(CUDNN_ACTIVATION_SIGMOID, x, similar(x))

function ∇sigmoid!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

function ∇sigmoid!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    ∇activation!(CUDNN_ACTIVATION_SIGMOID, y, gy, x, gx, beta=1.0)
end

"""
    tanh(x)
"""
function tanh(x::Var)
    y = tanh(x.data)
    df(gy) = hasgrad(x) && ∇tanh!(x.data, x.grad, y, gy)
    Var(y, [x], df)
end

tanh(x::CuArray) = activation!(CUDNN_ACTIVATION_TANH, x, similar(x))

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    ∇activation!(CUDNN_ACTIVATION_TANH, y, gy, x, gx, beta=1.0)
end
