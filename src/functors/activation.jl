export relu, sigmoid
import Base.tanh

"""
    relu(x::Var)

Rectifier liner unit.

## 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = relu(x)
```
"""
function relu(x::Var)
  y = relu(x.value)
  df(gy) = hasgrad(x) && ∇relu!(x.value, x.grad, y, gy)
  Var(y, nothing, df, [x])
end

"""
    sigmoid(x::Var)
"""
function sigmoid(x::Var)
  y = sigmoid(x.value)
  df(gy) = hasgrad(x) && ∇sigmoid!(x.value, x.grad, y, gy)
  Var(y, nothing, df, [x])
end

"""
    tanh(x::Var)
"""
function tanh(x::Var)
  y = tanh(x.value)
  df(gy) = hasgrad(x) && ∇tanh!(x.value, x.grad, y, gy)
  Var(y, nothing, df, [x])
end

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
