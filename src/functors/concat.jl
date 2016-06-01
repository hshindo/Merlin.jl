export concat

type Concat
  dim::Int
end

"""
Concatenate arrays along the given dimension.
- `concat(dim::Int, xs::Var...)`

## 👉 Example
```julia
x1 = Var(rand(Float32,7,5))
x2 = Var(rand(Float32,10,5))
y = concat(1, x1, x2)
```
"""
concat(dim::Int, xs::Var...) = forward0(Concat(dim), [xs...])
forward(f::Concat, args::Vector{Var}) = Var(concat(f.dim, map(a -> a.value, y.args)), f, args)
backward!(f::Concat, y::Var) = ∇concat!(f.dim, map(a -> a.grad, y.args), y.grad)

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
