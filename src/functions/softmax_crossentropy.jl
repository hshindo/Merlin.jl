export softmax_crossentropy

doc"""
    softmax_crossentropy(p::Var, x::Var, dim::Int)

Compute cross-entropy between $p$ and $x$ along the given dimension.

$ f(p,x)=-∑_{i} p_{i} \log x_{i} $

## Arguments
* p: var of `Vector{Int}` or `Matrix{Float}`. $p$ must be normalized.
* x: var of `Matrix{Float}`.

### 👉 Example
```julia
p = Var([1:5;])
x = Var(rand(Float32,10,5))
y = softmax_crossentropy(p, x)
```
"""
softmax_crossentropy(p::Var, x::Var, dim::Int) = SoftmaxCrossEntropy(dim)(p, x)

type SoftmaxCrossEntropy
  dim::Int
end

@compat function (f::SoftmaxCrossEntropy)(p::Var, x::Var)
  @checkargs f (p,x)
  @assert f.dim == 2
  logx = logsoftmax(x.value)
  y = softmax_crossentropy(p.value, logx)
  df(gy) = hasgrad(x) && ∇softmax_crossentropy!(p.value, logx, x.grad, gy)
  Var(y, df, [x])
end

function softmax_crossentropy{T}(p::Matrix{T}, logx::Matrix{T})
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

function softmax_crossentropy{T}(p::Vector{Int}, logx::Matrix{T})
  y = Array(T, 1, length(p))
  @inbounds @simd for j = 1:length(p)
    y[j] = -logx[p[j],j]
  end
  y
end

function ∇softmax_crossentropy!{T}(p::Matrix{T}, logx::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  for j = 1:size(p,2)
    g = gy[j]
    @inbounds @simd for i = 1:size(p,1)
      gx[i,j] += g * (exp(logx[i,j]) - p[i,j])
    end
  end
end

function ∇softmax_crossentropy!{T}(p::Vector{Int}, logx::Matrix{T}, gx::Matrix{T}, gy::Matrix{T})
  for j = 1:length(p)
    g = gy[j]
    @inbounds @simd for i = 1:size(logx,1)
      delta = ifelse(i == p[j], T(1), T(0))
      gx[i,j] += g * (exp(logx[i,j]) - delta)
    end
  end
end
