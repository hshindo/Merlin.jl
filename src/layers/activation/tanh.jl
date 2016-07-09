"""
    tanh(x)
"""
function tanh(x::Layer)
  l = Tanh(nothing, nothing, x)
  x.data == nothing || forward!(l)
  l
end

type Tanh <: Layer
  data
  grad
  x
end

tails(l::Tanh) = [l.x]
forward!(l::Tanh) = l.data = tanh(l.x.data)
backward!(l::Tanh) = hasgrad(l.x) && ∇tanh!(l.x.y, l.x.gy, l.y, l.gy)

tanh(x::CuArray) = activation!(CUDNN_ACTIVATION_TANH, x, similar(x))

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  ∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx, beta=1.0)
end
