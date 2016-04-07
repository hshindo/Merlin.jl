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

#=
function Base.call(f::Concat, args::Vector{Variable})
  xs = map(a -> a.value, args)
  y = concat(f.dim, xs)
  getgrad = gy -> ∇concat(f.dim, xs, gy)
  Variable(f, args, y, getgrad)
end
Base.call(f::Concat, args...) = call(f, [args...])
=#

function forward{T,N}(f::Concat, xs::Vector{Array{T,N}})
  sum = 0
  for x in xs
    sum += size(x, f.dim)
  end
  outsize = [size(xs[1])...]
  outsize[f.dim] = sum
  y = Array(T, outsize...)

  range = map(s -> 1:s, outsize)
  offset = 1
  for x in xs
    s = size(x, f.dim)
    range[f.dim] = offset:(offset + s - 1)
    y[range...] = x
    offset += s
  end
  y, gy -> backward(f, xs, gy)
end

function backward{T,N}(f::Concat, xs::Vector{Array{T,N}}, gy::Array{T,N})
  range = map(s -> 1:s, [size(gy)...])
  offset = 1
  map(xs) do x
    s = size(x, f.dim)
    range[f.dim] = offset:(offset + s - 1)
    offset += s
    gy[range...]
  end
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

function ∇concat{T,N}(dim::Int, xs::Vector{Array{T,N}}, gy::Array{T,N})
  range = map(s -> 1:s, [size(gy)...])
  offset = 1
  map(xs) do x
    s = size(x, dim)
    range[dim] = offset:(offset + s - 1)
    offset += s
    gy[range...]
  end
end

function ∇concat{T,N}(dim::Int, xs::Vector{CudaArray{T,N}}, gy::Array{T,N})

end
