type ReLU <: Functor
  x
  y
end

ReLU() = ReLU(nothing, nothing)

function forward!(f::ReLU, x)
  f.x = x
  f.y == nothing && (f.y = default(x))
  y = resize!(f.y, size(x))
  relu!(x.value, y.value)
  y
end

function relu!{T}(x::Array{T}, y::Array{T})
  for i = 1:length(x)
    xx = x[i]
    #y[i] = xx > T(0) ? xx : p * xx
    y[i] = xx > T(0) ? xx : T(0)
  end
end

function relu!(x::CudaArray, y::CudaArray)
  CUDNN.activation_forward(CUDNN.ACTIVATION_RELU, x, y)
end

backward!(f::ReLU) = ∇relu!(f.x, f.y)

function ∇relu!{T}(varx::Var{T}, vary::Var{T})
  varx.fixed && return
  x, gx = data(varx)
  y, gy = data(vary)
  for i = 1:length(x)
    #d = x[i] > T(0) ? gy[i] : p * gy[i]
    d = x[i] > T(0) ? gy[i] : T(0)
    gx[i] += d
  end
end

function ∇relu!{T}(varx::CudaVar{T}, vary::CudaVar{T})
  varx.fixed && return
  x, gx = data(varx)
  y, gy = data(vary)
  CUDNN.activation_backward(CUDNN.ACTIVATION_RELU, x, y)
end
