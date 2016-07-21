export concat

"""
    concat(dim::Int, xs::Var...)
    concat(dim::Int, xs::Vector{Var})

Concatenate arrays along the given dimension.
"""
function concat(dim::Int, xs::Vector{Var})
    y = concat(dim, map(x -> x.data, xs))
    df(gy) = ∇concat!(dim, map(x -> x.grad, xs), gy)
    Var(y, xs, df)
end
concat(dim::Int, xs::Var...) = concat(dim, Var[xs...])

function concat{T<:UniArray}(dim::Int, xs::Vector{T})
  sum = 0
  for x in xs
    sum += size(x,dim)
  end
  outsize = [size(xs[1])...]
  outsize[dim] = sum
  y = similar(xs[1], outsize...)

  range = map(s -> 1:s, outsize)
  offset = 1
  for x in xs
    s = size(x,dim)
    range[dim] = offset:(offset+s-1)
    y[range...] = x
    offset += s
  end
  y
end

function ∇concat!{T<:UniArray}(dim::Int, gxs::Vector{T}, gy::T)
  range = map(s -> 1:s, [size(gy)...])
  offset = 1
  for gx in gxs
    s = size(gx, dim)
    range[dim] = offset:(offset+s-1)
    BLAS.axpy!(eltype(gy)(1), gy[range...], gx)
    offset += s
  end
end
