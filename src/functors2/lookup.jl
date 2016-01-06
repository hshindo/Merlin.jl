type Lookup{K,V} <: Functor
  dict::Dict{K, Int}
  weights::Vector{Vector{V}}
  grads::Vector{Vector{V}}
  idset::Set{Int}
  readonly::Bool
  x
  y
end

function Lookup{K, V}(::Type{K}, ::Type{V}, outlength::Int)
  weights = Vector{V}[convert(Vector{V}, randn(outlength))]
  grads = Vector{V}[zeros(V, outlength)]
  Lookup(Dict{K, Int}(), weights, grads, Set{Int}(), false, nothing, nothing)
end

function Lookup{K,V}(path, ::Type{K}, ::Type{V})
  dict = Dict{K, Int}()
  weights = Vector{V}[]
  for line in open(readlines, path)
    items = split(chomp(line), '\t')
    key = K(items[1])
    id = get!(dict, key, length(dict)+1)
    vals = map(float, items[2:end])
    id <= length(weights) ? weights[id] = vals : push!(weights, vals)
  end
  grads = map(zeros, weights)
  Lookup(dict, weights, grads, Set{Int}(), true, nothing, nothing)
end

clone(f::Lookup) = Lookup(f.dict, f.weights, f.grads, f.idset, f.readonly, nothing, nothing)

function forward!(f::Lookup)
  y = lookup(f, f.x)
  f.y = Var(y)
end

function lookup{K,V}(f::Lookup{K,V}, x::Vector{K})
  y = Array(V, length(f.weights[1]), length(x))
  for i = 1:length(x)
    key = x[i]
    def = f.readonly ? 1 : length(f.dict) + 1 # TODO: must be fixed
    id = get!(f.dict, key, def)
    if id > length(f.weights)
      push!(f.weights, randn(size(y, 1)))
      push!(f.grads, zeros(size(y, 1)))
    end
    y[:, i] = f.weights[id]
  end
  y
end

function backward!(f::Lookup)
  ∇lookup!(f, f.x, f.y.grad)
end

function ∇lookup!{K,V}(f::Lookup{K,V}, x::Vector{K}, gy::Matrix{V})
  for i = 1:length(x)
    id = f.dict[x[i]]
    f.grads[id] += gy[:, i]
    union!(f.idset, id)
  end
  nothing
end

function optimize!(opt::Optimizer, l::Lookup)
  for id in l.idset
    update!(opt, l.weights[id], l.grads[id])
  end
  empty!(l.idset)
end
