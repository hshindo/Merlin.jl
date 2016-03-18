type Lookup <: Functor
  weights::Vector{Variable}
  idset::Set{Int}
end

Lookup(weights::Vector{Variable}) = Lookup(weights, Set{Int}())

function Lookup{T}(::Type{T}, insize::Int, outsize::Int)
  weights = Array(Variable, insize)
  for i = 1:insize
    w = convert(Vector{T}, randn(outsize))
    weights[i] = Variable(w, zeros(w))
  end
  Lookup(weights)
end

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

function forward!(f::Lookup, v::Variable)
  v.value = lookup(f, v[1].value)
  v.backward! = () -> ∇lookup!(f, v[1].value, v.grad)
end

function lookup(f::Lookup, x::Matrix{Int})
  ws = f.weights
  w1 = ws[1].value
  len = length(w1)
  T = eltype(w1)
  y = Array(T, len*size(x,1), size(x,2))
  offset = 1
  for i = 1:length(x)
    copy!(y, offset, ws[x[i]].value, 1, len)
    offset += len
  end
  y
end

function ∇lookup!{T}(f::Lookup, x::Matrix{Int}, gy::Matrix{T})
  ws = f.weights
  len = length(ws[1].value)
  size(x,1) > len && (gy = reshape(gy, len, length(x)))
  for i = 1:length(x)
    axpy!(1.0, gy[:, i], ws[x[i]].grad)
    union!(f.idset, x[i])
  end
end

function update!(opt::Optimizer, f::Lookup)
  for id in f.idset
    w = f.weights[id]
    update!(opt, w.value, w.grad)
  end
  empty!(f.idset)
end

type LookupLinear <: Functor
  w
  dict::Dict
end

function forward!()
end

function lookuplinear()
  if haskey(f.dict, id)
  else

  end
end
