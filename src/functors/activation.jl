export relu, sigmoid
import Base.tanh

type ReLU <: Functor; end

function forward{T<:Number}(f::ReLU, x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = max(x[i], T(0))
  end
  f, y
end

function forward(f::ReLU, x::CuArray)
  f, CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_RELU), x)
end

function backward!{T<:Number}(f::ReLU, x::Array{T}, gx::Array{T}, y, gy::Array{T})
  isempty(gx) && return
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
  end
end

function backward!(f::ReLU, x::CuArray, gx::CuArray, y, gy)
  isempty(gx) && return
  CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, dy, x, dx, beta=1.0)
end

"""
    relu(x)

Rectifier liner unit.
"""
relu(x::Var) = forward(ReLU(), x)


type Sigmoid <: Functor; end

function forward{T<:Number}(f::Sigmoid, x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = 1 / (1 + exp(-x[i]))
  end
  f, y
end

function forward(f::Sigmoid, x::CuArray)
  f, CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_SIGMOID), x)
end

function backward!{T<:Number}(f::Sigmoid, x, gx::Array{T}, y::Array{T}, gy::Array{T})
  isempty(gx) && return
  @inbounds @simd for i = 1:length(x)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end

function backward!(f::Sigmoid, x::CuArray, gx::CuArray, y, gy)
  isempty(gx) && return
  CUDNN.∇activation!(CUDNN_ACTIVATION_SIGMOID, y, dy, x, dx, beta=1.0)
end

"""
    sigmoid(x)
"""
sigmoid(x::Var) = forward(Sigmoid(), x)


type Tanh <: Functor; end

forward{T<:Number}(f::Tanh, x::Array{T}) = f, tanh(x)

function forward(f::Tanh, x::CuArray)
  f, CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_TANH), x)
end

function backward!{T<:Number}(f::Tanh, x, gx::Array{T}, y::Array{T}, gy::Array{T})
  isempty(gx) && return
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function backward!(f::Tanh, x::CuArray, gx::CuArray, y, gy)
  isempty(gx) && return
  CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx, beta=1.0)
end

"""
    tanh(x)
"""
tanh(x::Var) = forward(Tanh(), x)
