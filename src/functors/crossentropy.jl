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

@compat (f::CrossEntropy)(arg) = forward(f, arg)
function forward!(f::CrossEntropy, v::Variable)
  logq = logsoftmax(v[1].value)
  v.value = crossentropy(f.p, logq)
  v.backward! = () -> ∇crossentropy!(f.p, logq, v[1].grad, v.grad)
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
