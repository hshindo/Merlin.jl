type Concat <: Functor
  dim::Int
end

function forward!(f::Concat, v::Variable)
  xs = map(a -> a.value, v.args)
  v.value = cat(xs, f.dim)
end

function backward!(f::Concat, v::Variable)
  gy = v.grad
  offset = 0
  for i = 1:length(v.args)
    x = v[i].value
    s = size(x, f.dim)
    indices = AFArray([offset:offset+s-1])
    gx = lookup(gy, indices, f.dim)
    addgrad!(v[i], gx)
    offset += s
  end
end
