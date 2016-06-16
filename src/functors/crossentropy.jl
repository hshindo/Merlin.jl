export crossentropy

doc"""
    crossentropy(p, x::Var)

Compute cross-entropy between two distributions $p$ and $x$.

$ f(p,x)=-∑_{i} p_{i} \log x_{i} $

## Arguments
* p: `Vector{Int}` or `Matrix{Float}`. p must be normalized.
* x: var of `Matrix{Float}`.

### 👉 Example
```julia
p = [1:5;]
q = Var(rand(Float32,10,5))
y = crossentropy(p, x)
```
"""
crossentropy(p, x::Var) = forward(CrossEntropy(p,nothing), x)

type CrossEntropy
  p
  logx
end

@compat function (f::CrossEntropy)(x)
  logx = logsoftmax(x)
  y = crossentropy(f.p, logx)
  CrossEntropy(f.p,logx), y
end

function backward!(f::CrossEntropy, x, gx, y, gy::Array)
  ∇crossentropy!(f.p, f.logx, gx, gy)
end

function backward!(f::CrossEntropy, x, gx, y, gy::CuArray)
  throw("Not implemented yet.")
end

function crossentropy{T}(p::Matrix{T}, logx::Matrix{T})
  y = Array(T, 1, size(p,2))
  for j = 1:size(p,2)
    s = T(0)
    @inbounds @simd for i = 1:size(p,1)
      s += -p[i,j] * logx[i,j]
    end
    y[j] = s
  end
  y
end

function crossentropy{T}(p::Vector{Int}, logx::Matrix{T})
  y = Array(T, 1, length(p))
  @inbounds @simd for j = 1:length(p)
    y[j] = -logx[p[j],j]
  end
  y
end

function ∇crossentropy!{T}(p::Matrix{T}, logx::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  for j = 1:size(p,2)
    g = gy[j]
    @inbounds @simd for i = 1:size(p,1)
      gx[i,j] += g * (exp(logx[i,j]) - p[i,j])
    end
  end
end

function ∇crossentropy!{T}(p::Vector{Int}, logx::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  for j = 1:length(p)
    g = gy[j]
    @inbounds @simd for i = 1:size(logx,1)
      delta = ifelse(i == p[j], T(1), T(0))
      gx[i,j] += g * (exp(logx[i,j]) - delta)
    end
  end
end
