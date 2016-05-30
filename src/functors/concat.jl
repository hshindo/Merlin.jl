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
concat(dim::Int, xs::Var...) = Concat(dim)(xs)

@compat function (f::Concat)(xs::Var...)
  y = concat(f.dim, xs)
  backward!(y) = ∇concat!(f.dim, map(x -> x.grad, xs), y.grad)
  Var(y, f, xs, backward!)
end

function concat(dim::Int, xs::Tuple)
  sum = 0
  for x in xs
    sum += size(x.value, dim)
  end
  outsize = [size(xs[1].value)...]
  outsize[dim] = sum
  y = Array(T, outsize...)

  range = map(s -> 1:s, outsize)
  offset = 1
  for x in xs
    s = size(x.value, dim)
    range[dim] = offset:(offset+s-1)
    y[range...] = x.value
    offset += s
  end
  y
end

function ∇concat!(y::Var)
  range = map(s -> 1:s, [size(y.grad)...])
  offset = 1
  for a in y.args
    gx::Array = a.grad
    s = size(gx, dim)
    range[dim] = offset:(offset+s-1)
    BLAS.axpy!(T(1), gy[range...], gx)
    offset += s
  end
end

#=
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
=#
