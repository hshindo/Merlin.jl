export concat

type Concat <: Layer
  dim::Int
  xs::Vector{Layer}
  y
  gy
end

function concat(dim::Int, xs::Vector)
  y = any(x -> typeof(x.y) <: Symbol) ? Symbol() : concat(dim, map(x -> x.y, xs))
  Concat(dim, xs, y, nothing)
end

tails(l::Concat) = l.xs
backward!(l::Concat) = ∇concat!(l.dim, map(x -> x.grad, l.xs), l.gy)

function concat{T,N}(dim::Int, xs::Vector{Array{T,N}})
  sum = 0
  for x in xs
    sum += size(x, dim)
  end
  outsize = [size(xs[1])...]
  outsize[dim] = sum
  y = Array(T, outsize...)

  range = map(s -> 1:s, outsize)
  offset = 1
  for x in xs
    s = size(x, dim)
    range[dim] = offset:(offset+s-1)
    y[range...] = x
    offset += s
  end
  y
end

function ∇concat!{T,N}(dim::Int, gxs::Vector{Array{T,N}}, gy::Array{T,N})
  range = map(s -> 1:s, [size(gy)...])
  offset = 1
  for gx in gxs
    s = size(gx, dim)
    range[dim] = offset:(offset+s-1)
    BLAS.axpy!(T(1), gy[range...], gx)
    offset += s
  end
end
