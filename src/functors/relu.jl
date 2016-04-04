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

function call(f::ReLU, arg::Variable)
  y = relu(arg.value)
  backward! = () -> begin
    v[1].grad == nothing && (v[1].grad = zeros(v[1].value))
    ∇relu!(v[1].value, v[1].grad, v.grad)
  end
  Variable(f, [arg], y, backward!)
end

function forward!(f::ReLU, v::Variable)
  v.value = relu(v[1].value)
  v.backward! = () -> begin
    v[1].grad == nothing && (v[1].grad = zeros(v[1].value))
    ∇relu!(v[1].value, v[1].grad, v.grad)
  end
end

function relu{T,N}(x::Array{T,N})
  y = similar(x)
  @inbounds @simd for i = 1:length(x)
    xx = x[i]
    y[i] = xx > T(0) ? xx : T(0)
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
    d = x[i] > T(0) ? gy[i] : T(0)
    gx[i] += d
  end
end

#function ∇relu{T,N}(varx::CudaArray{T,N}, vary::CudaArray{T,N})
#  x, gx = data(varx)
#  y, gy = data(vary)
#  CUDNN.activation_backward(CUDNN.ACTIVATION_RELU, x, dx, y, dy)
#end
