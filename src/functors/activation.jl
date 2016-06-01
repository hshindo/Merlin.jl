export relu, sigmoid
import Base.tanh

type ReLU; end
type Sigmoid; end
type Tanh; end

for (t,f,df) in [(:ReLU,:relu,:∇relu!), (:Sigmoid,:sigmoid,:∇sigmoid!), (:Tanh,:tanh,:∇tanh!)]
  @eval begin
    $f(x::Var) = forward0($t(), x)
    forward(f::$t, args::Vector{Var}) = Var($f(args[1].value), nothing, f, args)
    backward!(f::$t, y::Var) = hasgrad(y[1]) && $df(y[1].value, y[1].grad, y.value, y.grad)
  end
end

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

function sigmoid{T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
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

tanh(x::CuArray) = CUDNN.activation!(ActivationDesc(CUDNN_ACTIVATION_TANH), x)

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx; beta=1.0)
end
