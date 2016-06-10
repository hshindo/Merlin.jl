import Base.max

type Max
  dim::Int
end

@compat function (f::Max)(args::Vector{Var})
  @checkargs f args
  x = args[1]
  y, idx = findmax(x.value, f.dim)
  df(gy) = hasgrad(x) && ∇max!(idx, x.grad, gy)
  Var(y, df, args)
end

"""
    max(x::Var, dim::Int)

Compute the maximum value along the given dimensions.
"""
max(x::Var, dim::Int) = Max(dim)([x])

function ∇max!{T,N}(idx::Array{Int,N}, gx::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end

function ∇max!{T,N}(idx, gx::CuArray{T,N}, gy::CuArray{T,N})
  throw("Not implemented yet.")
end
