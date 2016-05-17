export Concat

"""
Concatenate arrays along the given dimension.

## Functions
- `Concat(dim::Int)`

## 👉 Example
```julia
x1 = Var(rand(Float32,7,5))
x2 = Var(rand(Float32,10,5))
y = Concat(1)(x1, x2)
```
"""
type Concat <: Functor
  dim::Int
end

function forward(f::Concat, args::Vector{Var})
  y = concat(f.dim, map(a -> a.val, args))
  backward! = gy -> ∇concat!(f.dim, map(a -> a.grad, args), gy)
  Var(y, nothing, f, args, backward!)
end

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
