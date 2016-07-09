export sigmoid

"""
    sigmoid(x)
"""
function sigmoid(x::Layer)
  l = Sigmoid(nothing, nothing, x)
  x.data == nothing || forward!(l)
  l
end

type Sigmoid <: Layer
  data
  grad
  x
end

tails(l::Sigmoid) = [l.x]
forward!(l::Sigmoid) = l.data = sigmoid(l.x.data)
backward!(l::Sigmoid) = hasgrad(l.x) && ∇sigmoid!(l.x.y, l.x.gy, l.y, l.gy)

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
