export relu, sigmoid

"""
Rectifier Linear Unit.
"""
function relu(x::Var)
  typeof(x.value) == Symbol && return Var(nothing, relu, [x])
  Var(relu(x.value), relu, [x], ∇relu!)
end

function relu{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = max(x[i], T(0))
  end
  y
end

relu(x::CuArray) = CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_RELU), x)

function ∇relu!(y::Var)
  x = y[1]
  hasgrad(x) || return
  ∇relu!(x.value, x.grad, y.value, y.grad)
end

function ∇relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
  end
end

function ∇relu!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, dy, x, dx; beta=1.0)
end

""
function sigmoid(x::Var)
  typeof(x.value) == Symbol && return Var(x.value, sigmoid, [x])
  Var(sigmoid(x.value), sigmoid, [x], ∇sigmoid!)
end

function sigmoid{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    #y[i] = tanh(x[i]*0.5) * 0.5 + 0.5
    y[i] = 1 / (1 + exp(-x[i]))
  end
  y
end

sigmoid(x::CuArray) = CUDNN.activation!(ActivationDesc(CUDNN_ACTIVATION_SIGMOID), x)

function ∇sigmoid!(y::Var)
  x = y[1]
  hasgrad(x) || return
  ∇sigmoid!(x.value, x.grad, y.value, y.grad)
end

function ∇sigmoid!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end

function ∇sigmoid!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_SIGMOID, y, dy, x, dx; beta=1.0)
end

""
function Base.tanh(x::Var)
  typeof(x.value) == Symbol && return Var(x.value, tanh, [x])
  Var(tanh(x.value), tanh, [x], ∇tanh!)
end

Base.tanh(x::CuArray) = CUDNN.activation!(ActivationDesc(CUDNN_ACTIVATION_TANH), x)

function ∇tanh!(y::Var)
  x = y[1]
  hasgrad(x) || return
  ∇tanh!(x.value, x.grad, y.value, y.grad)
end

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx; beta=1.0)
end
