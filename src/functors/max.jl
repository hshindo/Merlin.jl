export Max

"""
## Max
Compute the maximum value of an array over the given dimensions.

### Functions
- `Max(dim::Int)`

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
f = Max(1)
y = f(x)
```
"""
type Max <: Functor
  dim::Int
end

@compat function (f::Max)(xs::Vector{Var})
  x = xs[1]
  y, idx = findmax(x.val, f.dim)
  backward! = gy -> hasgrad(x) && backward!(f, idx, x.grad, gy)
  Var(y, nothing, f, xs, backward!)
end
@compat (f::Max)(x::Var) = f([x])

function backward!{T,N}(f::Max, idx::Array{Int,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end
