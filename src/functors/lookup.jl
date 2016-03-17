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

function lookup(f::Lookup, x::Vector{Int})
  w = f.weights
  T = eltype(w[1].value)
  y = Array(T, length(w[1].value), length(x))
  for i = 1:length(x)
    y[:, i] = w[x[i]].value
  end
  y
end

function backward!(f::Lookup, v::Variable)
  ∇lookup!(f, v[1].value, v.grad)
end

function ∇lookup!{T}(f::Lookup, x::Vector{Int}, gy::Matrix{T})
  for i = 1:length(x)
    id = x[i]
    axpy!(1.0, gy[:, i], f.weights[id].grad)
    union!(f.idset, id)
  end
end

function update!(opt::Optimizer, f::Lookup)
  for id in f.idset
    w = f.weights[id]
    update!(opt, w.value, w.grad)
  end
  empty!(f.idset)
end
