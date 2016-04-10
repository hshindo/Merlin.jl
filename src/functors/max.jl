export Max

"""
## Max
Computes the maximum value of an array over the given dimensions.

### Functions
- `Max(dim::Int)`

### 👉 Example
```julia
x = Variable(rand(Float32,10,5))
f = Max(1)
y = f(x)
```
"""
type Max <: Functor
  dim::Int
end

function forward(f::Max, x)
  y, idx = findmax(x, f.dim)
  backward! = (gxs, gy) -> ∇max(idx, gxs[1], gy)
  #backward! = v -> ∇max(idx, v[1].grad, v.grad)
  y, backward!
end

function ∇max{T,N}(idx::Array{Int,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] = gy[i]
  end
end
