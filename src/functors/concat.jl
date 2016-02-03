type Concat <: Functor
  dim::Int
end

function forward!(f::Concat, v::Variable)
  xs = map(a -> a.value, v.args)
  v.value = cat(f.dim, xs)
end

function backward!(f::Concat, v::Variable)
  xs = map(a -> a.value, v.args)
  gxs = ∇concat(f.dim, xs, v.grad)
  for i = 1:length(v.args)
    addgrad!(v[i], gxs[i])
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
