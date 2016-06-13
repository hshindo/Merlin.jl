import Base.sum

type Sum
  dim::Int
end

forward{T<:Number}(f::Sum, x::Array{T}) = sum(x, f.dim)

function backward!{T}(f::Sum, x, gx, y, gy::Array{T})
  isempty(gx) && return
  throw("Not implemented yet.")
end

"""
    sum(x::Var, dim::Int)
"""
sum(x::Var, dim::Int) = forward(Sum(dim), x)
