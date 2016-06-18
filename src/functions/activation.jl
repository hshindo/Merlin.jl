export relu, sigmoid
import Base.tanh

"""
    relu(x)

Rectifier liner unit.
"""
function relu(x::Var)
  @checkargs relu (x,)
  y = relu(x.value)
  df(gy) = hasgrad(x) && ∇relu!(x.value, x.grad, y, gy)
  Var(y, df, [x])
end

function relu{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = max(x[i], T(0))
  end
  y
end

function relu(x::CuArray)
  activation!(CUDNN_ACTIVATION_RELU, x, similar(x))
end

function ∇relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
  end
end

function ∇relu!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, gy, x, gx, beta=1.0)
end

"""
    sigmoid(x)
"""
function sigmoid(x::Var)
  @checkargs sigmoid (x,)
  y = sigmoid(x.value)
  df(gy) = hasgrad(x) && ∇sigmoid!(x.value, x.grad, y, gy)
  Var(y, df, [x])
end

function sigmoid{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = 1 / (1 + exp(-x[i]))
  end
  y
end

function sigmoid(x::CuArray)
  activation!(CUDNN_ACTIVATION_SIGMOID, x, similar(x))
end

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
  @checkargs tanh (x,)
  y = tanh(x.value)
  df(gy) = hasgrad(x) && ∇tanh!(x.value, x.grad, y, gy)
  Var(y, df, [x])
end

function tanh(x::CuArray)
  activation!(CUDNN_ACTIVATION_TANH, x, similar(x))
end

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  ∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx, beta=1.0)
end
