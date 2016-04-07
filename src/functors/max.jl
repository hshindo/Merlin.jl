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
  y, idx = max(f.dim, x)
  backward = gy -> ∇max(idx, x, gy)
  y, backward
end

max{T,N}(dim::Int, x::Array{T,N}) = findmax(x, dim)

function ∇max{T,N}(idx::Array{Int,N}, x::Array{T,N}, gy::Array{T,N})
  gx = zeros(x)
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] = gy[i]
  end
  Array[gx]
end
