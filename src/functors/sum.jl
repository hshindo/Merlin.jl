import Base.sum

"""
    sum(x::Var, dim::Int)

Compute the sum along the given dimensions.
"""
sum(x::Var, dim::Int) = forward(Sum(dim), [x])

type Sum
  dim::Int
end

@compat function (f::Sum)(x::Var)
  @checkargs f (x,)
  y = sum(x.value, f.dim)
  df(gy) = hasgrad(x) && ∇sum!(x.grad, gy)
  Var(y, df, [x])
end

∇sum!(gx::Array, gy::Array) = broadcast!(.+, gx, gx, gy)

function ∇sum!(gx::CuArray, gy::CuArray)
  throw("Not implemented yet.")
end
