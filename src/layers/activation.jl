export relu, sigmoid
import Base.tanh

type Activation <: Layer
  mode
  x
  y
  gy
end

tails(l::Activation) = [x.x]

function backward!(l::Activation)
  df =
    l.mode == :relu ? ∇relu! :
    l.mode == :sigmoid ? ∇sigmoid! :
    l.mode == :tanh ? ∇tanh! : throw("Invalid mode.")
  hasgrad(l.x) && ∇relu!(l.x.y, l.x.gy, l.y, l.gy)
end

"""
    relu(x)

Rectifier liner unit.
"""
relu(x::Layer) = Activation(:relu, x, relu(x.y), nothing)
relu(x::GraphNode) = GraphNode(relu, x)

type ReLU <: Layer
  x
  y
  gy
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
sigmoid(x::Layer) = Sigmoid(x, sigmoid(x.y), nothing)
sigmoid(x::GraphNode) = GraphNode(sigmoid, x)

type Sigmoid <: Layer
  x
  y
  gy
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
