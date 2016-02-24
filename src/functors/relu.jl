type ReLU <: Functor
end

function forward!(f::ReLU, v::Variable)
  v.value = relu(v[1].value)
end

function relu{T,N}(x::Array{T,N})
  y = similar(x)
  for i = 1:length(x)
    xx = x[i]
    y[i] = xx > T(0) ? xx : T(0)
  end
  y
end

#function relu{T,N}(x::CudaArray{T,N})
#  y = alloc_gpu(T, size(x))
#  CUDNN.activation_forward(CUDNN.ACTIVATION_RELU, x, y)
#  y
#end

function backward!(f::ReLU, v::Variable)
  gx = ∇relu(v[1].value, v.grad)
  addgrad!(v[1], gx)
end

function ∇relu{T,N}(x::Array{T,N}, gy::Array{T,N})
  gx = similar(x)
  for i = 1:length(x)
    d = x[i] > T(0) ? gy[i] : T(0)
    gx[i] = d
  end
  gx
end

#function ∇relu{T,N}(varx::CudaArray{T,N}, vary::CudaArray{T,N})
#  x, gx = data(varx)
#  y, gy = data(vary)
#  CUDNN.activation_backward(CUDNN.ACTIVATION_RELU, x, dx, y, dy)
#end
