export concat

"""
    concat(dim::Int, xs::Var...)
    concat(dim::Int, xs::Vector{Var})

Concatenate arrays along the given dimension.
"""
concat(dim::Int, args::Vector{Var}) = forward(Concat(dim), args)
concat(dim::Int, args::Var...) = concat(dim, [args...])

type Concat
  dim::Int
end

@compat function (f::Concat){T,N}(xs::Vector{Array{T,N}})
  dim = f.dim
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
  f, y
end

@compat function (f::Concat){T<:CuArray}(xs::Vector{T})
  throw("Not implemented yet.")
end

function backward!{T,N}(f::Concat, xs, gxs::Vector{Array{T,N}}, y, gy::Array{T,N})
  dim = f.dim
  range = map(s -> 1:s, [size(gy)...])
  offset = 1
  for gx in gxs
    s = size(gx, dim)
    range[dim] = offset:(offset+s-1)
    BLAS.axpy!(T(1), gy[range...], gx)
    offset += s
  end
end
