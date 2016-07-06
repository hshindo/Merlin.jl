export relu, sigmoid
import Base.tanh

type Activation <: Layer
  x
  y
  df
  gy
end

tails(l::Activation) = [l.x]

backward!(l::Activation) = hasgrad(l.x) && l.df(l.x.y, l.x.gy, l.y, l.gy)

"""
    relu(x)

Rectifier liner unit.
"""
relu(x::Layer) = Activation(x, relu(x.y), ∇relu!, nothing)
relu(x::GraphNode) = GraphNode(relu, x)

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
sigmoid(x::Layer) = Activation(x, sigmoid(x.y), ∇sigmoid!, nothing)
sigmoid(x::GraphNode) = GraphNode(sigmoid, x)

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
tanh(x::Layer) = Activation(x, tanh(x.y), ∇tanh!, nothing)
tanh(x::GraphNode) = GraphNode(tanh, x)

tanh(x::CuArray) = activation!(CUDNN_ACTIVATION_TANH, x, similar(x))

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  ∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx, beta=1.0)
end
