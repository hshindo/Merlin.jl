import Base.max

"""
    max(x::Var, dim::Int)

Compute the maximum value along the given dimensions.
"""
max(x::Var, dim::Int) = forward(Max(dim,nothing), x)

type Max
  dim::Int
  idx
end

@compat function (f::Max){T}(x::Array{T})
  y, idx = findmax(x, f.dim)
  Max(f.dim,idx), y
end

function backward!{T}(f::Max, x, gx::Array{T}, y, gy::Array{T})
  isempty(gx) && return
  idx = f.idx::Vector{Int}
  @inbounds @simd for i = 1:length(idx)
    gx[idx[i]] += gy[i]
  end
end

function backward!(f::Max, x, gx::CuArray, y, gy::CuArray)
  throw("Not implemented yet.")
end
