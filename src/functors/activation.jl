export Activation

"""
## Activation
Activation function.

- `Activation(mode::AbstractString)`
    - mode: relu | tanh | sigmoid

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = Activation("relu")
y = f(x)
```
"""
type Activation <: Functor
  mode
end

function forward(f::Activation, args::Vector{Var})
  x = args[1]
  y = activation(f.mode, x.val)
  backward! = gy -> hasgrad(x) && ∇activation!(f.mode, x.val, x.grad, y, gy)
  Var(y, nothing, f, args, backward!)
end

function activation(mode::AbstractString, x)
  mode == "relu" && return relu(x)
  mode == "tanh" && return tanh(x)
  mode == "sigmoid" && return sigmoid(x)
  throw("Invalid mode: $(mode)")
end

function ∇activation!(mode::AbstractString, x, gx, y, gy)
  mode == "relu" && return ∇relu!(x, gx, y, gy)
  mode == "tanh" && return ∇tanh!(x, gx, y, gy)
  mode == "sigmoid" && return ∇sigmoid!(x, gx, y, gy)
  throw("Invalid mode: $(mode)")
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
    #y[i] = tanh(x[i]*0.5) * 0.5 + 0.5
    y[i] = 1 / (1 + exp(-x[i]))
  end
  y
end

relu(x::CuArray) = CUDNN.activation!(ActivationDesc(CUDNN_ACTIVATION_RELU), x, similar(x))
Base.tanh(x::CuArray) = CUDNN.activation!(ActivationDesc(CUDNN_ACTIVATION_TANH), x, similar(x))
sigmoid(x::CuArray) = CUDNN.activation!(ActivationDesc(CUDNN_ACTIVATION_SIGMOID), x, similar(x))

function ∇relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
  end
end

function ∇tanh!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function ∇sigmoid!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(x)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end

function ∇relu!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN.ACTIVATION_RELU, y, dy, x, dx; beta=1.0)
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN.ACTIVATION_TANH, y, dy, x, dx; beta=1.0)
end

function ∇sigmoid!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇activation!(CUDNN.ACTIVATION_SIGMOID, y, dy, x, dx; beta=1.0)
end
