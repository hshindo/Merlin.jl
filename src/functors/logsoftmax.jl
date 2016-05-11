export LogSoftmax, logsoftmax

"""
## LogSoftmax
Compute logarith of softmax function.

```math
f(x)=\frac{\exp(x_{i})}{\sum_{j}^{n}\exp(x_{j})},\;i=1,\ldots,n
```

### Functions
- `LogSoftmax()`

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = LogSoftmax()
y = f(x)
```
"""
type LogSoftmax <: Functor
end

function forward(f::LogSoftmax, args::Vector{Var})
  x = args[1]
  y = logsoftmax(x.val)
  backward! = gy -> hasgrad(x) && ∇logsoftmax!(x.val, y, x.grad, gy)
  Var(y, nothing, f, args, backward!)
end

function logsoftmax{T}(x::Matrix{T})
  y = similar(x)
  max = maximum(x, 1)
  for j = 1:size(x,2)
    sum = T(0)
    @inbounds @simd for i = 1:size(x,1)
      sum += exp(x[i,j] - max[j])
    end
    logz = log(sum)
    @inbounds @simd for i = 1:size(x,1)
      y[i,j] = x[i,j] - max[j] - logz
    end
  end
  y
end

function ∇logsoftmax!{T}(x::Matrix{T}, y::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  # d yj / d xi = delta(i=j) - exp(yi)
  for d = 1:size(x,2)
    for i = 1:size(x,1)
      expy = exp(y[i, d])
      for j = 1:size(x,1)
        delta = i == j ? T(1) : T(0)
        gx[i,d] += gy[j,d] * (delta - expy)
      end
    end
  end
end
