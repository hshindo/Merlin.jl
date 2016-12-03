import Base.tanh
export relu, sigmoid

"""
    relu(x::Var)

Rectifier function.
"""
function relu(x::Var)
    x.data == nothing && return Var(nothing, relu, (x,))
    y = relu(x.data)
    df(gy) = isconst(x) || ∇relu!(y, gy, x.data, x.grad)
    Var(y, relu, (x,), df)
end

function relu{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    y
end

function clipped_relu{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end
    y
end

relu(x::CuArray) = CUDNN.activation(CUDNN_ACTIVATION_RELU, x)

function ∇relu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
    end
end

function ∇clipped_relu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(T(0)<x[i]<T(20), gy[i], T(0))
    end
end

function ∇relu!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, gy, x, gx, beta=1.0)
end

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

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, gy, x, gx, beta=1.0)
end
