export crossentropy

type CrossEntropy
  logq
end

"""
    crossentropy(p::Var, q::Var)

Compute cross-entropy between two distributions \$p\$ and \$q\$,
where \$p\$ is usually correct labels and \$q\$ is predicted values.

```math
f(p,q)=-∑_{x} p_{x} \log q_{x}
```

## Arguments
* \$p\$: variable of `Vector{Int}` or `Matrix{Float}`.
If \$p\$ is `Matrix{Float}`, it must be normalized.
* \$q\$: variable of `Matrix{Float}`.

### 👉 Example
```julia
p = Var([1:5;])
q = Var(rand(Float32,10,5))
y = crossentropy(p, q)
```
"""
crossentropy(p::Var, q::Var) = init(CrossEntropy(nothing), [p,q])

function forward(f::CrossEntropy, args::Vector{Var})
  p, q = args[1], args[2]
  logq = logsoftmax(q.value)
  y = crossentropy(p.value, logq)
  CrossEntropy(logq), y
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
