export Concat

"""
## Concat
Concatenates arrays along the given dimension.

### Functions
- `Concat(dim::Int)`

### 👉 Example
```julia
x1 = Variable(rand(Float32,7,5))
x2 = Variable(rand(Float32,10,5))
f = Concat(1)
y = f(x1, x2) # or y = f([x1,x2])
```
"""
type Concat <: Functor
  dim::Int
end

function forward(f::Concat, xs::Vector)
  y = concat(f.dim, xs)
  backward! = (gxs, gy) -> ∇concat!(f.dim, gxs, gy)
  #backward! = v -> ∇concat!(f.dim, map(a -> a.grad, v.args), v.grad)
  y, backward!
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

function concat{T,N}(dim::Int, xs::Vector{CudaArray{T,N}})

end

function ∇concat!{T,N}(dim::Int, gxs::Vector{Array{T,N}}, gy::Array{T,N})
  range = map(s -> 1:s, [size(gy)...])
  offset = 1
  for gx in gxs
    s = size(gx, dim)
    range[dim] = offset:(offset + s - 1)
    copy!(gx, gy[range...])
    offset += s
  end
end

function ∇concat!{T,N}(dim::Int, xs::Vector{CudaArray{T,N}}, gy::CudaArray{T,N})

end
