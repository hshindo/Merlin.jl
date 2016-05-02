export Sigmoid

"""
## Sigmoid

- `Sigmoid()`

### 👉 Example
```julia
x = rand(Float32,10,5)
f = Sigmoid()
y = f(x)
```
"""
type Sigmoid <: Functor
end

@compat (f::Sigmoid)(arg) = forward(f, arg)
function forward!(f::Sigmoid, v::Variable)
  v.value = tanh(v[1].value * 0.5) * 0.5 + 0.5
  v.backward! = () -> hasgrad(v[1]) && ∇sigmoid!(v.value, v[1].grad, v.grad)
end

sigmoid(x::Variable) = Sigmoid()(x)

function ∇sigmoid!{T,N}(y::Array{T,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(y)
    gx[i] += gy[i] * y[i] * (T(1) - y[i])
  end
end
