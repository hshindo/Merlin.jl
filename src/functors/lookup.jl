type AFLookup <: Functor
  weight::Variable
  idset::Set{Int}
end

function AFLookup{T}(::Type{T}, xlength::Int, ylength::Int)
  w = randn(AFArray{T}, xlength, ylength) |> Variable
  AFLookup(w, Set{Int}())
end

forward!(f::AFLookup, v::Variable) = v.value = aflookup(f.weight.value, v[1].value)

aflookup(weight::AFMatrix, indices::AFVector{Int}) = ArrayFire.lookup(weight, indices, 2)




type Lookup <: Functor
  params::Vector{Variable}
  idset::Set{Int}
end

Lookup(params::Vector{Variable}) = Lookup(params, Set{Int}())

function Lookup{T}(::Type{T}, xlength::Int, ylength::Int)
  params = Array(Variable, xlength)
  for i = 1:xlength
    params[i] = convert(Vector{T}, randn(ylength)) |> Variable
  end
  Lookup(params)
end

function Lookup{T}(path, ::Type{T})
  lines = open(readlines, path)
  params = Array(Variable, length(lines))
  for i = 1:length(lines)
    items = split(chomp(lines[i]), ' ')
    v = map(x -> parse(T,x), items)
    params[i] = Variable(v)
  end
  Lookup(params)
end

forward!(f::Lookup, v::Variable) = v.value = lookup(f, v[1].value)

function lookup(f::Lookup, x::Vector{Int})
  p = f.params
  y = Array(eltype(p[1].value), length(p[1].value), length(x))
  for i = 1:length(x)
    y[:, i] = p[x[i]].value
  end
  y
end

function backward!(f::Lookup, v::Variable)
  ∇lookup!(f, v[1].value, v.grad)
end

function ∇lookup!(f::Lookup, x::Vector{Int}, gy::Matrix)
  for i = 1:length(x)
    id = x[i]
    addgrad!(f.params[id], gy[:, i])
    union!(f.idset, id)
  end
end

function optimize!(opt::Optimizer, f::Lookup)
  for id in f.idset
    p = f.params[id]
    update!(opt, p.value, p.grad)
  end
  empty!(f.idset)
end
