export Softmax, softmax

"""
## Softmax
\$ f(x)=\frac{\exp(x_{i})}{\sum_{j}^{n}\exp(x_{j})},\;i=1,\ldots,n \$

### Functions
- `Softmax()`

### 👉 Example
```julia
x = rand(Float32,10,5)
f = Softmax()
y = f(x)
```
"""
type Softmax <: Functor
end

@compat (f::Softmax)(arg) = forward(f, arg)
function forward!(f::Softmax, v::Variable)
  v.value = softmax(v[1].value)
  v.backward! = () -> hasgrad(v[1]) && ∇softmax2!(v[1].value, v.value, v[1].grad, v.grad)
end

function softmax{T}(x::Matrix{T})
  y = similar(x)
  max = maximum(x, 1)
  for j = 1:size(x,2)
    z = T(0)
    @inbounds @simd for i = 1:size(x,1)
      z += exp(x[i,j] - max[j])
    end
    z == T(0) && error("z == 0")
    @inbounds @simd for i = 1:size(x,1)
      y[i,j] = exp(x[i,j] - max[j]) / z
    end
  end
  y
end

function ∇softmax!{T}(x::Matrix{T}, y::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  # d yi / d xj = yi * (delta (i=j) - yj)
  g = y .* gy
  sumdx = sum(g, 1)
  g -= y .* sumdx
  copy!(gx, g)
end

function ∇softmax2!{T}(x::Matrix{T}, y::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  # d yj / d xi = yj * (delta (i=j) - yi)
  for d = 1:size(x,2)
    for i = 1:size(x,1)
      yi = y[i,d]
      for j = 1:size(x,1)
        delta = i == j ? T(1) : T(0)
        gx[i,d] += gy[j,d] * y[j,d] * (delta - yi)
      end
    end
  end
end