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

function call(f::Max, arg::Variable)
  y, idx = max(f.dim, arg.value)
  backward! = (gxs, gy) -> ∇max!(idx, gxs[1], gy)
  Variable(f, [arg], y, backward!)
end

function forward!(f::Max, v::Variable)
  y, idx = max(f.dim, v[1].value)
  v.value = y
  v.backward! = () -> begin
    v[1].grad == nothing && (v[1].grad = zeros(v[1].value))
    ∇max!(idx, v[1].grad, v.grad)
  end
end

function max{T,N}(dim::Int, x::Array{T,N})
  findmax(x, dim)
end

function ∇max!{T,N}(idx::Array{Int,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end
