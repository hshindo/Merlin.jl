export relu

"""
    relu(x)

Rectifier liner unit.
"""
function relu(x::Layer)
  l = ReLU(nothing, nothing, x)
  x.data == nothing || forward!(l)
  l
end

type ReLU <: Layer
  data
  grad
  x
end

tails(l::ReLU) = [l.x]
forward!(l::ReLU) = l.data = relu(l.x.data)
backward!(l::ReLU) = hasgrad(l.x) && ∇relu!(l.x.data, l.x.grad, l.data, l.grad)

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
