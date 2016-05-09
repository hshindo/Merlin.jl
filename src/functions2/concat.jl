export Concat

"""
## Concat
Concatenate arrays along the given dimension.

### Functions
- `Concat(dim::Int)`

### 👉 Example
```julia
x1 = Variable(rand(Float32,7,5))
x2 = Variable(rand(Float32,10,5))
f = Concat(1)
y = f(x1, x2)
```
"""
type Concat <: Functor
  dim::Int
end

@compat function (f::Concat)(args::Variable...)
  xs = map(a -> a.val, args)
  y = concat(f.dim, xs...)
  backward! = gy -> begin
    offset = 1
    range = map(s -> 1:s, [size(gy)...])
    for a in args
      hasgrad(a) || continue
      s = size(a.grad, f.dim)
      range[f.dim] = offset:(offset + s - 1)
      axpy!(T(1), gy[range...], gx)
      offset += s
    end
  end
  Variable(f, args, y, backward!)
end

function forward(f::Concat, args::Variable...)
  xs = map(a -> a.val, args)
  y = concat(f.dim, xs...)
  backward! = gy -> begin
    offset = 1
    range = map(s -> 1:s, [size(gy)...])
    for a in args
      hasgrad(a) || continue
      s = size(a.grad, f.dim)
      range[f.dim] = offset:(offset + s - 1)
      axpy!(T(1), gy[range...], gx)
      offset += s
    end
  end
  Variable(f, args, y, backward!)
end

function concat{T,N}(dim::Int, xs::Tuple{Vararg{Array{T,N}}})
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
