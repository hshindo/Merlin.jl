export relu, sigmoid

type ReLU
end

relu(x::Var) = forward(ReLU(), [x])
forward!(f::ReLU, y::Var) = y.value = relu(y[1].value)
backward!(f::ReLU, y::Var) = hasgrad(y[1]) && ∇relu!(y[1].value, y[1].grad, y.value, y.grad)

function relu{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = max(x[i], T(0))
  end
  y
end

relu(x::CuArray) = CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_RELU), x)

function ∇relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
  end
end

function ∇relu!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, dy, x, dx; beta=1.0)
end

type Sigmoid
end

sigmoid(x::Var) = forward(Sigmoid(), [x])
forward!(f::Sigmoid, y::Var) = y.value = sigmoid(y[1].value)
backward!(f::Sigmoid, y::Var) = hasgrad(y[1]) && ∇sigmoid!(y[1].value, y[1].grad, y.value, y.grad)

function sigmoid{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    #y[i] = tanh(x[i]*0.5) * 0.5 + 0.5
    y[i] = 1 / (1 + exp(-x[i]))
  end
  y
end

sigmoid(x::CuArray) = CUDNN.activation!(ActivationDesc(CUDNN_ACTIVATION_SIGMOID), x)

function ∇sigmoid!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end

function ∇sigmoid!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_SIGMOID, y, dy, x, dx; beta=1.0)
end

type Tanh
end

Base.tanh(x::Var) = forward(Tanh(), [x])
forward!(f::Tanh, y::Var) = y.value = tanh(y[1].value)
backward!(f::Tanh, y::Var) = hasgrad(y[1]) && ∇tanh!(y[1].value, y[1].grad, y.value, y.grad)

Base.tanh(x::CuArray) = CUDNN.activation!(ActivationDesc(CUDNN_ACTIVATION_TANH), x)

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx; beta=1.0)
end
