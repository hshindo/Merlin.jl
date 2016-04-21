export Tanh

"""
## Tanh

- `Tanh()`

### 👉 Example
```julia
x = rand(Float32,10,5)
f = Tanh()
y = f(x)
```
"""
type Tanh <: Functor
end

@compat (f::Tanh)(arg) = forward(f, arg)
function forward!(f::Tanh, v::Variable)
  v.value = tanh(v[1].value)
  v.backward! = () -> hasgrad(v[1]) && ∇tanh!(v.value, v[1].grad, v.grad)
end

function ∇tanh!{T,N}(y::Array{T,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx)
    gx[i] += gy[i] * (T(1) - y[i] * y[i])
  end
end
