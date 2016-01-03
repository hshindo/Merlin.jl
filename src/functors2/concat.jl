type Concat <: Functor
  dim::Int
  x
  y
end

Concat(dim::Int) = Concat(dim, nothing, nothing)

function getysize{T<:AbstractVar}(dim::Int, xs::Vector{T})
  sum = 0
  for x in xs
    sum += size(x, dim)
  end
  ysize = [size(xs[1])...]
  ysize[dim] = sum
  tuple(ysize)
end

function forward!(f::Concat)
  xs = map(x -> x.value, f.x)
  ysize = getysize(f.dim, f.x)
  f.y == nothing && (f.y = default(x[1]))
  y = resize!(f.y, ysize)
  concat!(f.dim, xs, y.value)
end

function concat!{T,N}(dim::Int, xs::Vector{Array{T,N}}, y::Array{T,N})
  range = map(s -> 1:s, [size(y)...])
  index = 1
  for x in xs
    s = size(x, dim)
    range[dim] = index:(index + s - 1)
    y[range...] = x
    index += s
  end
end

function backward!(f::Concat)
  gxs = map(v -> v.grad, f.x)
  ∇concat!(f.dim, gxs, f.y.grad)
end

function ∇concat!(dim::Int, gxs::Vector{Array{T,N}}, gy::Array{T,N})
  range = map(s -> 1:s, [size(gy)...])
  index = 1
  for gx in gxs
    s = size(gx, dim)
    range[dim] = index:(index + s - 1)
    axpy!(T(1), gy[range...], gx)
    index += s
  end
end
