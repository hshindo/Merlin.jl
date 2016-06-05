export concat

type Concat
  dim::Int
end

@compat function (f::Concat)(xs::Vector{Var})
  y = concat(f.dim, map(x -> x.value, xs))
  df(gy) = ∇concat!(f.dim, map(x -> x.grad, xs), gy)
  Var(y, df, xs)
end

"""
    concat(dim::Int, xs::Var...)
    concat(dim::Int, xs::Vector{Var})

Concatenate arrays along the given dimension.

## 👉 Example
```julia
x1 = Var(rand(Float32,7,5))
x2 = Var(rand(Float32,10,5))
y = concat(1, x1, x2)
```
"""
concat(dim::Int, xs::Var...) = concat(dim, [xs...])
concat(dim::Int, xs::Vector{Var}) = forward(Concat(dim), xs)

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
