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

@compat function (f::Concat)(args) = forward(f, args)
@compat function (f::Concat)(args...) = f(args)

function forward{T}(xs::Vector{Var{T}})
  val = concat(f.dim, map(x -> x.val, xs))
  backward! = y -> backward!(f, xs, y)
  Variable(val, f, xs, backward!)
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
    range[dim] = offset:(offset + s - 1)
    y[range...] = x
    offset += s
  end
  y
end

function backward!{T<:Array}(f::Concat, xs::Vector{Var{T}}, y::Var{T})
  gy = get(y.grad)
  offset = 1
  range = map(s -> 1:s, [size(gy)...])
  for a in args
    isnull(a.grad) && continue
    gx = get(a.grad)
    s = size(gx, f.dim)
    range[f.dim] = offset:(offset + s - 1)
    axpy!(eltype(gy)(1), gy[range...], gx)
    offset += s
  end
end
