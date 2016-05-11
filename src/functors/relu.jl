export ReLU

"""
## ReLU
Rectifier linear unit.

- `ReLU()`

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = ReLU()
y = f(x)
```
"""
type ReLU <: Functor
end

@compat function (f::ReLU)(xs::Vector{Var})
  x = xs[1]
  y = relu(x.val)
  backward! = gy -> hasgrad(x) && ∇relu!(x.val, x.grad, gy)
  Var(y, nothing, f, xs, backward!)
end
@compat (f::ReLU)(x::Var) = f([x])

function relu{T,N}(x::Array{T,N})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    y[i] = max(x[i], T(0))
  end
  y
end

function relu{T,N}(x::CudaArray{T,N})
  y = similar(x)
  activation_forward!(CUDNN_ACTIVATION_RELU, 1.0, x, 0.0, y)
  y
end

function ∇relu!{T,N}(x::Array{T,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(x)
    gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
  end
end

function ∇relu(x::CudaArray, gx::CudaArray, y::CudaArray, gy::CudaArray)
  CUDNN.activation_backward(CUDNN.ACTIVATION_RELU, x, dx, y, dy)
end
