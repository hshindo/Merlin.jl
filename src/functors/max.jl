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

function forward(f::Max, args::Vector{Var})
  isnothing(args) && return Var()
  x = args[1]
  y, idx = findmax(x.val, f.dim)
  backward! = gy -> hasgrad(x) && backward!(f, idx, x.grad, gy)
  Var(y, nothing, f, args, backward!)
end

function backward!{T,N}(f::Max, idx::Array{Int,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end
