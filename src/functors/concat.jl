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
    indices = AFArray([offset:offset+s])
    gx = lookup(x, indices, f.dim)
    addgrad!(v[i], gx)
    offset += s
  end
end

function ∇concat{T,N}(dim::Int, xs::Vector{Array{T,N}}, gy::Array{T,N})
  range = map(s -> 1:s, [size(gy)...])
  index = 1
  gxs = Array(Array{T,N}, length(xs))
  for i = 1:length(xs)
    x = xs[i]
    s = size(x, dim)
    range[dim] = index:(index + s - 1)
    gxs[i] = gy[range...]
    index += s
  end
  gxs
end
