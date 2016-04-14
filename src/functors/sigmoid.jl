export Sigmoid

"""
## Sigmoid

- `Sigmoid()`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Sigmoid()
y = f(x)
```
"""
type Sigmoid <: Functor
end

@compat (f::Sigmoid)(arg) = forward(f, arg)
function forward!(f::Sigmoid, v::Variable)
  v.value = tanh(v[1].value * 0.5) * 0.5 + 0.5
  v.backward! = () -> ∇sigmoid!(v.value, v[1].grad, v.grad)
end

function ∇sigmoid!{T,N}(y::Array{T,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(y)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end
