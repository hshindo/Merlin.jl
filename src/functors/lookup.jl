export Lookup

"""
## Lookup
Lookup variables.

### Functions
- Lookup(insize::Int, outsize::Int)

### 👉 Example
```julia
x = Variable([1:5])
f = Lookup(Float32,10000,100)
y = f(x)
```
"""
type Lookup <: Functor
  #weights::Vector{Variable}
  ws::Vector
  gws::Vector
  idset::Set{Int}
end

#Lookup(weights::Vector{Variable}) = Lookup(weights, Set{Int}())
Lookup{T}(ws::Vector{T}) = Lookup(ws, map(zeros, ws), Set{Int}())

"""
- T: Type
- insize::Int
- outsize::Int
"""
function Lookup{T}(::Type{T}, insize::Int, outsize::Int)
  ws = Array(Vector{T}, insize)
  for i = 1:insize
    ws[i] = convert(Vector{T}, randn(outsize))
  end
  Lookup(ws)
end
#=
function Lookup{T}(::Type{T}, insize::Int, outsize::Int)
  ws = Array(T, insize)
  for i = 1:insize
    w = convert(Vector{T}, randn(outsize))
    ws[i] = Variable(w, zeros(w))
  end
  Lookup(weights)
end
=#

"""
- path: initial values
- T::Type
"""
function Lookup{T}(path, ::Type{T})
  lines = open(readlines, path)
  ws = Array(Vector{T}, length(lines))
  for i = 1:length(lines)
    items = split(chomp(lines[i]), ' ')
    ws[i] = map(x -> parse(T,x), items)
  end
  Lookup(ws)
end
#=
function Lookup{T}(path, ::Type{T})
  lines = open(readlines, path)
  weights = Array(Variable, length(lines))
  for i = 1:length(lines)
    items = split(chomp(lines[i]), ' ')
    v = map(x -> parse(T,x), items)
    weights[i] = Variable(v, zeros(v))
  end
  Lookup(weights)
end
=#


function forward(f::Lookup, x::Array{Int})
  y = lookup(f, x)
  backward = gy -> ∇lookup(f, x, gy)
  y, backward
end

function lookup(f::Lookup, x::Matrix{Int})
  w1 = f.ws[1]
  len = length(w1)
  T = eltype(w1)
  y = Array(T, len*size(x,1), size(x,2))
  offset = 1
  for i = 1:length(x)
    copy!(y, offset, f.ws[x[i]], 1, len)
    offset += len
  end
  y
end

function ∇lookup{T}(f::Lookup, x::Matrix{Int}, gy::Matrix{T})
  len = length(f.ws[1])
  offset = 1
  for i = 1:length(x)
    gw = f.gws[x[i]]
    axpy!(len, T(1.0), pointer(gy,offset), stride(gy,1), pointer(gw), stride(gw,1))
    offset += len
    union!(f.idset, x[i])
  end
  [nothing]
end

function update!(opt::Optimizer, f::Lookup)
  for id in f.idset
    w, gw = f.ws[id], f.gws[id]
    update!(opt, w, gw)
  end
  empty!(f.idset)
end
