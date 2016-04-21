export Lookup

"""
## Lookup
Lookup variables.

### Functions
- Lookup(insize::Int, outsize::Int)

### 👉 Example
```julia
x = [1:5]
f = Lookup(Float32,10000,100)
y = f(x)
```
"""
type Lookup <: Functor
  weights::Vector{Variable}
  idset::Set{Int}
end

Lookup(weights::Vector) = Lookup(weights, Set{Int}())

"""
- T: Type
- insize::Int
- outsize::Int
"""
function Lookup{T}(::Type{T}, insize::Int, outsize::Int)
  weights = Array(Variable, insize)
  for i = 1:insize
    w = convert(Vector{T}, randn(outsize))
    weights[i] = Variable(w, zeros(w))
  end
  Lookup(weights)
end

"""
- path: initial values
- T::Type
"""
function Lookup{T}(path, ::Type{T})
  lines = open(readlines, path)
  weights = Array(Variable, length(lines))
  for i = 1:length(lines)
    items = split(chomp(lines[i]), ' ')
    w = map(x -> parse(T,x), items)
    weights[i] = Variable(w, zeros(w))
  end
  Lookup(weights)
end

@compat (f::Lookup)(arg) = forward(f, arg)
function forward!(f::Lookup, v::Variable)
  v.value = lookup(f, v[1].value)
  v.backward! = () -> ∇lookup!(f, v[1].value, v.grad)
end

function lookup(f::Lookup, x::Matrix{Int})
  w1 = f.weights[1].value
  len = length(w1)
  T = eltype(w1)
  y = Array(T, len*size(x,1), size(x,2))
  offset = 1
  for i = 1:length(x)
    copy!(y, offset, f.weights[x[i]].value, 1, len)
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
    union!(f.idset, x[i])
  end
end

function update!(opt::Optimizer, f::Lookup)
  for id in f.idset
    update!(opt, f.weights[id].value, f.weights[id].grad)
  end
  empty!(f.idset)
end
