export Softmax, softmax

"""
## Softmax

```math
f(x)=\frac{\exp(x_{i})}{\sum_{j}^{n}\exp(x_{j})},\;i=1,\ldots,n
```

```math
p(x) = {\\exp(f(x)) \\over \\sum_{x_2} \\exp(f(x))}
```

### Functions
- `Softmax()`

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = Softmax()
y = f(x)
```
"""
type Softmax <: Functor
end

function forward(f::Softmax, args::Vector{Var})
  x = args[1]
  y = softmax(x.val)
  backward! = gy -> hasgrad(x) && ∇softmax!(x.grad, y, gy)
  Var(y, nothing, f, args, backward!)
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

function softmax(x::CuArray)
  y = similar(x)
  CUDNN.softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, x, y)
end

function ∇softmax2!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  # d yi / d xj = yi * (delta (i=j) - yj)
  g = y .* gy
  sumdx = sum(g, 1)
  g -= y .* sumdx
  copy!(gx, g)
end

function ∇softmax!{T}(gx::Matrix{T}, y::Matrix{T}, gy::Matrix{T})
  # d yj / d xi = yj * (delta (i=j) - yi)
  for d = 1:size(gx,2)
    for i = 1:size(gx,1)
      yi = y[i,d]
      for j = 1:size(gx,1)
        delta = i == j ? T(1) : T(0)
        gx[i,d] += gy[j,d] * y[j,d] * (delta - yi)
      end
    end
  end
end

function ∇softmax!(gx::CuArray, y::CuArray, gy::CuArray)
  CUDNN.∇softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, y, gy, gx; beta=1.0)
end
