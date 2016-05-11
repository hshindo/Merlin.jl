export CrossEntropy

"""
## CrossEntropy
Compute cross-entropy between a true distribution: \$p\$ and the target distribution: \$q_x\$.
\$p\$ and \$q\$ are assumed to be normalized (sums to one).
To noamalize 'Var's, use `LogSoftmax`.

```math
f(x;p)=-∑_{x} p \log q_{x}
```

### Functions
- `CrossEntropy()`

### 👉 Example
```julia
p = Var([1:5])
x = Var(rand(Float32,10,5))
f = CrossEntropy()
y = f(p, q)
```
"""
type CrossEntropy <: Functor
end

@compat function (f::CrossEntropy)(xs::Vector{Var})
  p, q = xs[1], xs[2]
  logq = logsoftmax(q.val)
  y = crossentropy(p.val, logq)
  backward! = gy -> hasgrad(q) && ∇crossentropy!(p.val, logq, q.grad, gy)
  Var(y, nothing, f, xs, backward!)
end
@compat (f::CrossEntropy)(xs...) = f([xs...])

function crossentropy{T}(p::Matrix{T}, logq::Matrix{T})
  y = Array(T, 1, size(p,2))
  for j = 1:size(p,2)
    s = T(0)
    @inbounds @simd for i = 1:size(p,1)
      s += -p[i,j] * logq[i,j]
    end
    y[j] = s
  end
  y
end

function crossentropy{T}(p::Vector{Int}, logq::Matrix{T})
  y = Array(T, 1, length(p))
  @inbounds @simd for j = 1:length(p)
    y[j] = -logq[p[j],j]
  end
  y
end

function ∇crossentropy!{T}(p::Matrix{T}, logq::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  for j = 1:size(p,2)
    g = gy[j]
    @inbounds @simd for i = 1:size(p,1)
      gx[i,j] += g * (exp(logq[i,j]) - p[i,j])
    end
  end
end

function ∇crossentropy!{T}(p::Vector{Int}, logq::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  for j = 1:length(p)
    g = gy[j]
    @inbounds @simd for i = 1:size(logq,1)
      delta = ifelse(i == p[j], T(1), T(0))
      gx[i,j] += g * (exp(logq[i,j]) - delta)
    end
  end
end
