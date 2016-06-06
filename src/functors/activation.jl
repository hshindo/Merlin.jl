export relu, sigmoid
import Base.tanh

type ReLU; end
type Sigmoid; end
type Tanh; end

@compat function (f::ReLU)(args::Vector{Var})
  x = args[1]
  y = relu(x.value)
  df(gy) = hasgrad(x) && ∇relu!(x.value, x.grad, y, gy)
  Var(y, df, args)
end

@compat function (f::Sigmoid)(args::Vector{Var})
  x = args[1]
  y = sigmoid(x.value)
  df(gy) = hasgrad(x) && ∇sigmoid!(x.value, x.grad, y, gy)
  Var(y, df, args)
end

@compat function (f::Tanh)(args::Vector{Var})
  x = args[1]
  y = tanh(x.value)
  df(gy) = hasgrad(x) && ∇tanh!(x.value, x.grad, y, gy)
  Var(y, df, args)
end

"""
    relu(x)

Rectifier liner unit.
"""
relu(x::Var) = forward(ReLU(), [x])

"""
    sigmoid(x)
"""
sigmoid(x::Var) = forward(Sigmoid(), [x])

"""
    tanh(x)
"""
tanh(x::Var) = forward(Tanh(), [x])

function relu{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = max(x[i], T(0))
  end
  y
end

function sigmoid{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = 1 / (1 + exp(-x[i]))
  end
  y
end

relu(x::CuArray) = CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_RELU), x)
sigmoid(x::CuArray) = CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_SIGMOID), x)
tanh(x::CuArray) = CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_TANH), x)

function ∇relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
  end
end

function ∇sigmoid!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function ∇relu!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, dy, x, dx, beta=1.0)
end

function ∇sigmoid!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_SIGMOID, y, dy, x, dx, beta=1.0)
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx, beta=1.0)
end
