type Max
  dim::Int
  idx
end

"""
Compute the maximum value along the given dimensions.

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = max(x, 1)
```
"""
Base.max(dim::Int, x::Var) = forward(Max(dim,nothing), [x])

function forward!(f::Max, y::Var)
  y.value, idx = findmax(y[1].value, f.dim)
  y.f = Max(f.dim, idx)
end

backward!(f::Max, y::Var) = hasgrad(y[1]) && ∇max!(f.idx, y[1].grad, y.grad)

function ∇max!{T,N}(idx::Array{Int,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end
