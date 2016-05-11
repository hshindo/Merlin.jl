export Lookup

"""
## Lookup
Lookup variables.

### Functions
- Lookup(insize::Int, outsize::Int)

### 👉 Example
```julia
x = Var([1:5])
f = Lookup(Float32,10000,100)
y = f(x)
```
"""
type Lookup <: Functor
  weights::Vector{Var}
end

Lookup{T}(weights::Vector{Vector{T}}) = Lookup(map(w -> Var(w,zeros(w)), weoghts))

function Lookup{T}(::Type{T}, insize::Int, outsize::Int)
  weights = Array(Var, insize)
  for i = 1:insize
    w = convert(Vector{T}, randn(outsize))
    weights[i] = Var(w, grad=zeros(w))
  end
  Lookup(weights)
end

function Lookup{T}(path, ::Type{T})
  lines = open(readlines, path)
  weights = Array(Var, length(lines))
  for i = 1:length(lines)
    items = split(chomp(lines[i]), ' ')
    w = map(x -> parse(T,x), items)
    weights[i] = Variable(w, zeros(w))
  end
  Lookup(weights)
end

function (f::Lookup, args::Vector{Var})
  x = args[1]
  y = lookup(f, x.val)
  args = [x]
  for id in x.val
    push!(args, f.weights[id])
  end
  backward! = gy -> ∇lookup!(f, x.val, gy)
  Var(y, nothing, f, args, backward!)
end

function lookup(f::Lookup, x::Matrix{Int})
  w1 = f.weights[1].val
  len = length(w1)
  T = eltype(w1)
  y = Array(T, len*size(x,1), size(x,2))
  offset = 1
  for i = 1:length(x)
    copy!(y, offset, f.weights[x[i]].val, 1, len)
    offset += len
  end
  y
end

function ∇lookup!{T}(f::Lookup, x::Matrix{Int}, gy::Matrix{T})
  len = length(f.weights[1].value)
  offset = 1
  for i = 1:length(x)
    gw = f.weights[x[i]].grad
    BLAS.axpy!(len, T(1), pointer(gy,offset), stride(gy,1), pointer(gw), stride(gw,1))
    offset += len
    #union!(f.idset, x[i])
  end
end
