export CrossEntropy

"""
## CrossEntropy
Computes cross-entropy between a true distribution \$p\$ and the target distribution \$q
\$p\$ and \$q\$ are assumed to be normalized (sums to one).
To noamalize 'Variable's, use `Softmax`.

```math
f(x;p)=-∑_{x} p \log q_{x}
```

### Functions
- `CrossEntropy(p::Matrix)`
- `CrossEntropy(p::Vector{Int})`

### 👉 Example
```julia
p = [1:5]
f = CrossEntropy()
x = Variable(rand(Float32,10,5))
y = f(x)
```
"""
type CrossEntropy <: Functor
end

@compat (f::CrossEntropy)(args) = forward(f, args)

function forward(f::CrossEntropy, args::Variable...)
  y = cross(f.p, arg.val)
  backward! = gy -> begin
    hasgrad(arg) && ∇crossentropy!(p, logq, v[2].grad, v.grad)
  end
  Variable(y, f, args, backward!)
end

function cross{T}(p::Matrix{T}, q::Matrix{T})
  y = Array(T, 1, size(p,2))
  for j = 1:size(p,2)
    s = T(0)
    @inbounds @simd for i = 1:size(p,1)
      s += -p[i,j] * log(q[i,j])
    end
    y[j] = s
  end
  y
end

function backward!(f::CrossEntropy)
end

function forward!(f::CrossEntropy, v::Variable)
  p = v[1].value
  logq = logsoftmax(v[2].value)
  v.value = crossentropy(p, logq)
  v.backward! = () -> hasgrad(v[2]) && ∇crossentropy!(p, logq, v[2].grad, v.grad)
end

function forward{T,N}(f::CrossEntropy, arg::Variable)
  p = arg.value
  backward! = v -> begin

  end
  Variable(y, y, f, [arg], backward!)
end



#=
type CrossEntropy <: Functor
  p
end

@compat (f::CrossEntropy)(arg) = forward(f, arg)
function forward!(f::CrossEntropy, v::Variable)
  logq = logsoftmax(v[1].value)
  v.value = crossentropy(f.p, logq)
  v.backward! = () -> hasgrad(v[1]) && ∇crossentropy!(f.p, logq, v[1].grad, v.grad)
end
=#

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
