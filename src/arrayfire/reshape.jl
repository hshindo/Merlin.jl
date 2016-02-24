type Reshape <: Functor
  dims
end

Reshape(dims::Int...) = Reshape(dims)

function forward!(f::Reshape, v::Variable)
  x = v[1].value
  v.value = reshape(x, f.dims)
end

function backward!(f::Reshape, v::Variable)
  x = v[1].value
  gx = reshape(v.grad, size(x))
  addgrad!(v[1], gx)
end
