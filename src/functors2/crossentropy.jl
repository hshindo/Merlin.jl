export crossentropy

"""
Compute cross-entropy between a source distribution: \$p\$ and the target distribution: \$q\$,
where \$q\$ is assumed to be output of `logsoftmax`.

```math
f(p,q)=-∑_{x} p_{x} \log q_{x}
```
## Arguments
- `p::Var`: source distribution
- `logq::Var`: target distribution

### 👉 Example
```julia
p = Var([1:5])
x = Var(rand(Float32,10,5))
logq = softmax(x, mode=:log)
y = crossentropy(p, logq)
```
"""
crossentropy(p::Var, logq::Var) = forward0(CrossEntropy(), [p,logq])

type CrossEntropy <: Functor
end

function forward(f::CrossEntropy, args::Vector{Var})
  p, logq = args[1], args[2]
  y = crossentropy(p.val, logq.val)
  backward! = gy -> hasgrad(q) && ∇crossentropy!(p.val, logq, q.grad, gy)
  Var(y, nothing, f, args, backward!)
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
