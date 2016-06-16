export relu, sigmoid
import Base.tanh

"""
    relu(x)

Rectifier liner unit.
"""
relu(x::Var) = forward(ReLU(), x)

type ReLU; end

@compat function (f::ReLU){T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = max(x[i], T(0))
  end
  f, y
end

@compat function (f::ReLU)(x::CuArray)
  f, CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_RELU), x)
end

function backward!{T}(f::ReLU, x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  isempty(gx) && return
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
  end
end

function backward!{T<:CuArray}(f::ReLU, x::T, gx::T, y::T, gy::T)
  isempty(gx) && return
  CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, gy, x, gx, beta=1.0)
end


"""
    sigmoid(x)
"""
sigmoid(x::Var) = forward(Sigmoid(), x)

type Sigmoid; end

@compat function (f::Sigmoid){T}(x::Array{T})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = 1 / (1 + exp(-x[i]))
  end
  f, y
end

@compat function (f::Sigmoid)(x::CuArray)
  f, CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_SIGMOID), x)
end

function backward!{T}(f::Sigmoid, x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  isempty(gx) && return
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end

function backward!{T<:CuArray}(f::Sigmoid, x::T, gx::T, y::T, gy::T)
  isempty(gx) && return
  CUDNN.∇activation!(CUDNN_ACTIVATION_SIGMOID, y, gy, x, gx, beta=1.0)
end


"""
    tanh(x)
"""
tanh(x::Var) = forward(Tanh(), x)

type Tanh; end

@compat (f::Tanh)(x::Array) = f, tanh(x)

@compat function (f::Tanh)(x::CuArray)
  f, CUDNN.activation(ActivationDesc(CUDNN_ACTIVATION_TANH), x)
end

function backward!{T}(f::Tanh, x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
  isempty(gx) && return
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end

function backward!{T<:CuArray}(f::Tanh, x::T, gx::T, y::T, gy::T)
  isempty(gx) && return
  CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, dy, x, dx, beta=1.0)
end
