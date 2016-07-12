import Base.sum

type Sum <: Var
  data
  grad
  tails::Vector{Var}
  dim::Int
end

"""
    sum(x, dim::Int)

Compute the sum along the given dimensions.
"""
function sum(x::Var, dim::Int)
  y = hasdata(x) ? sum(x.data,dim) : nothing
  Sum(y, nothing, [x], dim)
end
@compat (v::Sum)(x::Var) = sum(x, v.dim)

backward!(v::Sum) = hasgrad(v[1]) && ∇sum!(v[1].grad, v.grad)

∇sum!(gx::Array, gy::Array) = broadcast!(.+, gx, gx, gy)

function ∇sum!(gx::CuArray, gy::CuArray)
  throw("Not implemented yet.")
end
