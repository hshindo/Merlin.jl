type MaxPool2D <: Functor
  dim::Int
end

function forward!(f::MaxPool2D, v::Variable)
  x = v[1].value
  y, ind = findmax(x, f.dim)
  v.value = y
  v.work = ind
end
