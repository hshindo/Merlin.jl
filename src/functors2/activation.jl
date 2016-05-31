export relu, sigmoid

abstract Activation <: Functor

type ReLU <: Activation
end
type Tanh <: Activation
end
type Sigmoid <: Activation
end

function forward(f::Activation, args::Vector{Var})
  x = args[1]
  y = activation(f, x.val)
  backward! = gy -> hasgrad(x) && ∇activation!(f, x.val, x.grad, y, gy)
  Var(y, nothing, f, args, backward!)
end

""
relu(x::Var) = forward0(ReLU(), [x])

""
Base.tanh(x::Var) = forward0(Tanh(), [x])

""
sigmoid(x::Var) = forward0(Sigmoid(), [x])

function activation{T}(f::ReLU, x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = max(x[i], T(0))
  end
  y
end
activation(f::ReLU, x::CuArray) = CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_RELU), x)

activation(f::Tanh, x) = tanh(x.val)
activation(f::Tanh, x::CuArray) = CUDNN.activation!(ActivationDesc(CUDNN_ACTIVATION_TANH), x)

function activation{T}(f::Sigmoid, x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    #y[i] = tanh(x[i]*0.5) * 0.5 + 0.5
    y[i] = 1 / (1 + exp(-x[i]))
  end
  y
end
activation(f::Sigmoid, x::CuArray) = CUDNN.activation!(ActivationDesc(CUDNN_ACTIVATION_SIGMOID), x)

function ∇activation!{T}(f::ReLU, x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
  end
end
function ∇activation!(f::ReLU, x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, dy, x, dx; beta=1.0)
end

function ∇activation!{T}(f::Tanh, x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end
function ∇activation!(f::Tanh, x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx; beta=1.0)
end

function ∇activation!{T}(f::Sigmoid, x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end
function ∇activation!(f::Sigmoid, x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_SIGMOID, y, dy, x, dx; beta=1.0)
end
