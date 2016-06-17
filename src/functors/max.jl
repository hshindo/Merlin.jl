import Base.max

"""
    max(x::Var, dim::Int)

Compute the maximum value along the given dimensions.
"""
max(x::Var, dim::Int) = Max(dim)(x)

type Max
  dim::Int
end

@compat function (f::Max)(x::Var)
  @checkargs f (x,)
  y, idx = findmax(x.value, f.dim)
  df(gy) = hasgrad(x) && ∇max!(idx, x.grad, gy)
  Var(y, df, [x])
end

function ∇max!{T}(idx::Vector{Int}, gx::Array{T}, gy::Array{T})
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end
