export Tanh

"""
## Tanh

- `Tanh()`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Tanh()
y = f(x)
```
"""
type Tanh <: Functor
end

function call(f::Tanh, arg::Variable)
  y = tanh(arg.value)
  getgrad = gy -> ∇tanh(y, gy)
  Variable(f, [arg], y, getgrad)
end

function ∇tanh{T,N}(y::Array{T,N}, gy::Array{T,N})
  gx = similar(y)
  @inbounds @simd for i = 1:length(gx)
    gx[i] = gy[i] * (T(1) - y[i] * y[i])
  end
  [gx]
end
