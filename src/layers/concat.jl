type Concat <: Layer
  value
  grad
  dim::Int
  deps::Vector{Layer}
end

function Concat(dim::Int, deps::Vector{Layer})
  xs = map(deps, x -> x.value)
  value = concat(dim, xs)
  Concat(value, nothing, dim, deps)
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

function gradient(l::Concat)

end
