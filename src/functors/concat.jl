export Concat

"""
## Concat
Concatenate arrays along the given dimension.

### Functions
- `Concat(dim::Int)`

### 👉 Example
```julia
x1 = Var(rand(Float32,7,5))
x2 = Var(rand(Float32,10,5))
f = Concat(1)
y = f(x1, x2)
```
"""
type Concat <: Functor
  dim::Int
end

@compat (f::Concat)(args) = forward(f, args)
@compat (f::Concat)(args...) = forward(f, args)

function forward(f::Concat, vars::Vector{Var})
  xs = map(v -> v.val, vars)
  y = concat(f.dim, xs)
  backward! = gy -> ∇concat!(f.dim, map(v -> v.grad, vars), gy)
  Var(y, f, vars, backward!)
end
forward(f::Concat, args::Tuple{Vararg{Var}}) = forward(f, Var[args...])

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
    range[dim] = offset:(offset + s - 1)
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
    axpy!(T(1), gy[range...], gx)
    offset += s
  end
end
