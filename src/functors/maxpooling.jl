type MaxPooling <: Functor
  dim::Int
end

function forward!(f::MaxPooling, v::Variable)
  x = v[1].value
  y = maximum(x, f.dim)
  v.value = y
end

function backward!(f::MaxPooling, v::Variable)
  x = v[1]
  cond = x.value >= v.value
  gx = v.grad .* cond
  addgrad!(x, gx)
end
