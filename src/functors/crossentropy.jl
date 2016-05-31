export crossentropy

type CrossEntropy
  logq
end

"""
Compute cross-entropy between two distributions \$p\$ and \$q\$,
where \$p\$ is correct labels and \$q\$ is predicted un-normalized scores.
- \$p\$: `Var` of `Vector{Int}` or `Matrix`.
- \$q\$: `Var` of `Matrix`.

```math
f(p,q)=-∑_{x} p_{x} \log q_{x}
```

### 👉 Example
```julia
p = Var([1:5;])
q = Var(rand(Float32,10,5))
y = crossentropy(p, q)
```
"""
crossentropy(p::Var, q::Var) = forward(CrossEntropy(nothing), [p,q])

function forward!(f::CrossEntropy, y::Var)
  p, q = y[1], y[2]
  logq = logsoftmax(q.value)
  y.value = crossentropy(p.value, logq)
  y.f = CrossEntropy(logq)
end

function backward!(f::CrossEntropy, y::Var)
  p, q = y[1], y[2]
  hasgrad(p) && throw("Not implemented yet.")
  hasgrad(q) && ∇crossentropy!(p.value, f.logq, q.grad, y.grad)
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
      delta = ifelse(i == p[j], T(1), T(0))
      gq[i,j] += g * (exp(logq[i,j]) - delta)
    end
  end
end
