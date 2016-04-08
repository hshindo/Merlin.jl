export ReLU

"""
## ReLU
Rectifier linear unit.

- `ReLU()`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = ReLU()
y = f(x)
```
"""
type ReLU <: Functor
end

function forward(f::ReLU, x)
  y = relu(x)
  backward = gy -> ∇relu(x, gy)
  y, backward
end

function relu{T,N}(x::Array{T,N})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = Base.max(x[i], T(0))
  end
  y
end

function relu{T,N}(x::CudaArray{T,N})
  y = similar(x)
  activation_forward!(CUDNN_ACTIVATION_RELU, 1.0, x, 0.0, y)
  y
end

function ∇relu{T,N}(x::Array{T,N}, gy::Array{T,N})
  gx = similar(x)
  @inbounds @simd for i = 1:length(x)
    gx[i] = ifelse(x[i]>T(0), gy[i], T(0))
  end
  Array[gx]
end

#function ∇relu{T,N}(varx::CudaArray{T,N}, vary::CudaArray{T,N})
#  x, gx = data(varx)
#  y, gy = data(vary)
#  CUDNN.activation_backward(CUDNN.ACTIVATION_RELU, x, dx, y, dy)
#end
