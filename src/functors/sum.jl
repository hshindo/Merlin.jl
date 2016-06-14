import Base.sum

type Sum
  dim::Int
end

@compat function (f::Sum)(args::Vector{Var})
  x = args[1]
  y = sum(x.value, f.dim)
  df(gy) = hasgrad(x) && ∇sum!(x.grad, gy)
  Var(y, df, [x])
end

"""
    sum(x::Var, dim::Int)

Compute the sum along the given dimensions.
"""
sum(x::Var, dim::Int) = forward(Sum(dim), [x])

function ∇sum!{T,N}(gx::Array{T,N}, gy::Array{T,N})
  broadcast!(+, gx, gx, gy)
end

function ∇sum!{T,N}(gx::CuArray{T,N}, gy::CuArray{T,N})
  throw("Not implemented yet.")
end
