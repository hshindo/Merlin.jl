type Reshape{N} <: Functor
  dims::NTuple{N,Int}
end

Reshape(dims::Int...) = Reshape(dims)

function forward!(f::Reshape, v::Variable)
  x = v[1].value
  y = reshape(x, f.dims)
  v.value = y
end
