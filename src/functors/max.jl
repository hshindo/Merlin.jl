export Max

"""
## Max
Computes the maximum value of an array over the given dimensions.

### Functions
- `Max(dim::Int)`

### 👉 Example
```julia
x = rand(Float32,10,5)
f = Max(1)
y = f(x)
```
"""
type Max <: Functor
  dim::Int
end

@compat (f::Max)(arg) = forward(f, arg)

function forward(f::Max, v::Variable)
  y, idx = findmax(v.val, f.dim)
  backward! = gy -> hasgrad(v) && backward!(f, idx, v.grad, gy)
  Variable(y, [v], backward!)
end

function forward!(f::Max, v::Variable)
  y, idx = findmax(v[1].val, f.dim)
  v.val = y
  v.backward! = () -> hasgrad(v[1]) && backward!(f, idx, v[1].grad, v.grad)
end

function backward!{T,N}(f::Max, idx::Array{Int,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end
