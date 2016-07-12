export concat

type Concat <: Var
  data
  grad
  tails::Vector{Var}
end

"""
    concat(dim::Int, xs::Var...)
    concat(dim::Int, xs::Vector{Var})

Concatenate arrays along the given dimension.
"""
function concat(dim::Int, xs::Vector)
  y = concat(dim, map(x -> x.data, xs))
  Concat(y, nothing, xs, dim)
end
function concat(dim::Int, xs::Var...)
  all(hasdata, xs) && return concat(dim, Var[xs...])
  Concat(nothing, nothing, xs, dim)
end
@compat (v::Concat)(xs::Var...) = concat(v.dim, Var[xs...])

backward!(v::Concat) = ∇concat!(v.dim, map(x -> x.grad, v.xs), v.grad)

function concat{T<:Array}(dim::Int, xs::Vector{T})
  sum = 0
  for x in xs
    sum += size(x, dim)
  end
  outsize = [size(xs[1])...]
  outsize[dim] = sum
  y = similar(xs[1], outsize...)

  range = map(s -> 1:s, outsize)
  offset = 1
  for x in xs
    s = size(x, dim)
    range[dim] = offset:(offset+s-1)
    y[range] = x
    offset += s
  end
  y
end

function ∇concat!{T<:Array}(dim::Int, gxs::Vector{T}, gy::T)
  range = map(s -> 1:s, [size(gy)...])
  offset = 1
  for gx in gxs
    s = size(gx, dim)
    range[dim] = offset:(offset+s-1)
    BLAS.axpy!(T(1), sub(gy,range...), gx)
    offset += s
  end
end
