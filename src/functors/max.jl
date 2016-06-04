import Base.max

"""
    max(x::Var, dim::Int)

Compute the maximum value along the given dimensions.

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = max(x, 1)
```
"""
function max(x::Var, dim::Int)
  y, idx = findmax(x.value, dim)
  f(gy) = hasgrad(x) && ∇max!(idx, x.grad, gy)
  Var(y, nothing, f, [x])
end

function ∇max!{T,N}(idx::Array{Int,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end

function ∇max!{T,N}(idx, gx::CuArray{T,N}, gy::CuArray{T,N})
  throw("Not implemented yet.")
end
