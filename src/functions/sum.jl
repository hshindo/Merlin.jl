import Base.sum

type Sum
  dim::Int
end

@compat function (f::Sum)(args::Vector{Var})
  @checkargs f args
  x = args[1]
  throw("Not implemented yet.")
end

"""
    sum(x::Var, dim::Int)
"""
sum(x::Var, dim::Int) = Sum(dim)([x])
