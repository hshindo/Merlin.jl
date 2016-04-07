export CrossEntropy

"""
## CrossEntropy
Computes cross-entropy between a true distribution \$p\$ and the target distribution \$q\$.

\$ f(x; p) = -\sum_{x} p \log q_x \$

### Functions
- `CrossEntropy(p::Matrix)`
- `CrossEntropy(p::Vector{Int})`

### 👉 Example
```julia
p = Variable([1:10])
f = CrossEntropy(p)
x = Variable(rand(Float32,50,10))
y = f(x)
```
"""
type CrossEntropy <: Functor
  p
end

function forward(f::CrossEntropy, x::Array)
  logq = logsoftmax(x)
  y = crossentropy(f.p, logq)
  backward = gy -> ∇crossentropy(f.p, logq, gy)
  y, backward
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

function ∇crossentropy{T}(p::Matrix{T}, logq::Matrix{T}, gy::Matrix{T})
  gq = similar(logq)
  for j = 1:size(p,2)
    g = gy[j]
    @inbounds @simd for i = 1:size(p,1)
      gq[i,j] = g * (exp(logq[i,j]) - p[i,j])
    end
  end
  Array[gq]
end

function ∇crossentropy{T}(p::Vector{Int}, logq::Matrix{T}, gy::Matrix{T})
  gq = similar(logq)
  for j = 1:length(p)
    g = gy[j]
    @inbounds @simd for i = 1:size(logq,1)
      delta = i == p[j] ? T(1) : T(0)
      gq[i,j] = g * (exp(logq[i,j]) - delta)
    end
  end
  Array[gq]
end
