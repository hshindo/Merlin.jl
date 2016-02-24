type Concat <: Functor
  dim::Int
end

function forward!(f::Concat, v::Variable)
  xs = map(a -> a.value, v.args)
  v.value = concat(xs, f.dim)
end

function concat(xs::Vector{AFArray}, dim::Int)
  cat(xs, dim)
end

function concat{T,N}(xs::Vector{Array{T,N}}, dim::Int)
  sum = 0
  for x in xs
    sum += size(x, dim)
  end
  outsize = [size(xs[1])...]
  outsize[dim] = sum
  y = Array(T, outsize...)

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

function backward!(f::Concat, v::Variable)
  gy = v.grad
  offset = 0
  for i = 1:length(v.args)
    x = v[i].value
    s = size(x, f.dim)
    indices = AF.range(eltype(x), (s,)) + offset
    gx = lookup(gy, indices, f.dim)
    addgrad!(v[i], gx)
    offset += s
    #addgrad!(v[i], zeros(x))
  end
end
