import Base.sum

"""
    sum(x, dim::Int)

Compute the sum along the given dimensions.
"""
sum(x::Layer, dim::Int) = Sum(dim, x, sum(x.y,dim), nothing)
sum(x::GraphNode, dim::Int) = GraphNode(sum, x, dim)

type Sum <: Layer
  dim::Int
  x
  y
  gy
end

tails(l::Sum) = [l.x]

backward!(l::Sum) = hasgrad(l.x) && ∇sum!(l.x.gy, l.gy)

∇sum!(gx::Array, gy::Array) = broadcast!(.+, gx, gx, gy)

function ∇sum!(gx::CuArray, gy::CuArray)
  throw("Not implemented yet.")
end
