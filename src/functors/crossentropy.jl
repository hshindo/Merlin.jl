export crossentropy

type CrossEntropy; end

@compat function (f::CrossEntropy)(args::Vector{Var})
  p, q = args[1], args[2]
  logq = logsoftmax(q.value)
  y = crossentropy(p.value, logq)
  df(gy) = hasgrad(q) && ∇crossentropy!(p.value, logq, q.grad, gy)
  Var(y, df, [q])
end

doc"""
    crossentropy(p, q)

Compute cross-entropy between two distributions $p$ and $q$.

$ f(p,q)=-∑_{x} p_{x} \log q_{x} $

## Arguments
* p: var of `Vector{Int}` or `Matrix{Float}`. p must be normalized.
* q: var of `Matrix{Float}`.

### 👉 Example
```julia
p = Var([1:5;])
q = Var(rand(Float32,10,5))
y = crossentropy(p, q)
```
"""
crossentropy(p::Var, q::Var) = forward(CrossEntropy(), [p,q])

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

function ∇crossentropy!{T}(p::Matrix{T}, logq::Matrix{T}, gq::Matrix{T}, gy::Matrix{T})
  for j = 1:size(p,2)
    g = gy[j]
    @inbounds @simd for i = 1:size(p,1)
      gq[i,j] += g * (exp(logq[i,j]) - p[i,j])
    end
  end
end

function ∇crossentropy!{T}(p::Vector{Int}, logq::Matrix{T}, gq::Matrix{T}, gy::Matrix{T})
  for j = 1:length(p)
    g = gy[j]
    @inbounds @simd for i = 1:size(logq,1)
      delta = ifelse(i == p[j], T(1), T(0))
      gq[i,j] += g * (exp(logq[i,j]) - delta)
    end
  end
end
