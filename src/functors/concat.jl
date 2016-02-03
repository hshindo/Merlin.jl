type Concat <: Functor
  dim::Int
end

function forward!(f::Concat, v::Variable)
  xs = map(a -> a.value, v.args)
  v.value = concat(f.dim, xs)
end

function concat{T,N}(dim::Int, xs::Vector{Array{T,N}})
  sum = 0
  for x in xs
    sum += size(x, dim)
  end
  outsize = [size(xs[1])...]
  outsize[dim] = sum
  y = alloc_cpu(T, outsize...)

  range = map(s -> 1:s, outsize)
  index = 1
  for x in xs
    s = size(x, dim)
    range[dim] = index:(index + s - 1)
    y[range...] = x
    index += s
  end
  y
end

concat{T,N}(dim::Int, xs::Vector{AFArray{T,N}}) = cat(dim, xs)

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
