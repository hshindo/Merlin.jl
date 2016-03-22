export CrossEntropy

"""
## 🔨 CrossEntropy
Computes cross-entropy between a true distribution \(p\) and the target distribution \(q\).
\$\$f(p,q)=-\sum_{x}p(x)\log q(x)\$\$

### Functions
- `CrossEntropy(p::Matrix)`

### 👉 Example
```julia
#p = Variable(rand(Float32,10,5))
#f = CrossEntropy(p)
#q = Variable(rand(Float32,10,5))
#y = f(q)
```
"""
type CrossEntropy <: Functor
  p
end

function forward!(f::CrossEntropy, v::Variable)
  logq = logsoftmax(v[1].value)
  v.value = crossentropy(f.p, logq)
  v.backward! = () -> begin
    v[1].grad == nothing && (v[1].grad = zeros(v[1].value))
    ∇crossentropy!(f.p, logq, v[1].grad, v.grad)
  end
end

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
      delta = i == p[j] ? T(1) : T(0)
      gq[i,j] += g * (exp(logq[i,j]) - delta)
    end
  end
end
