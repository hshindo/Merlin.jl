type Max
  dim::Int
end

@compat function (f::Max)(x::Var)
  y, idx = findmax(x.value, f.dim)
  function ∇max!(y)
    x = y[1]
    hasgrad(x) && ∇max!(idx, x.grad, y.grad)
  end
  Var(y, f, [x], ∇max!)
end

"""
Compute the maximum value along the given dimensions.

### 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = max(x, 1)
```
"""
Base.max(x::Var, dim::Int) = Max(dim)(x)

function ∇max!{T,N}(idx::Array{Int,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end
