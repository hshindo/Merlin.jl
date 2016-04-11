export Softmax

"""
## Softmax
\$ f(x)=\frac{\exp(x_{i})}{\sum_{j}^{n}\exp(x_{j})},\;i=1,\ldots,n \$

### Functions
- `Softmax()`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Softmax()
y = f(x)
```
"""
type Softmax <: Functor
end

@compat (f::Softmax)(arg) = forward(f, arg)
function forward!(f::Softmax, v::Variable)
  v.value = softmax(v[1].value)
  v.backward! = () -> ∇softmax!(v[1].value, v.value, v[1].grad, v.grad)
end

function ∇softmax!{T}(x::Matrix{T}, y::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  # d(y_j) / d(x_i) = delta(i = j) - exp(y_i)
  for d = 1:size(x,2)
    for i = 1:size(x,1)
      expy = exp(y[i, d])
      for j = 1:size(x,1)
        delta = i == j ? T(1.0) : T(0.0)
        gx[i, d] += gy[j, d] * (delta - expy)
      end
    end
  end
end
