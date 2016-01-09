type Lookup{K,V} <: Functor
  dict::Dict{K, Int}
  weights::Vector{Vector{V}}
  grads::Vector{Vector{V}}
  idset::Set{Int}
  readonly::Bool
end

function Lookup{K, V}(::Type{K}, ::Type{V}, outlength::Int)
  weights = Vector{V}[convert(Vector{V}, randn(outlength))]
  grads = Vector{V}[zeros(V, outlength)]
  Lookup(Dict{K,Int}(), weights, grads, Set{Int}(), false)
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
  Lookup(dict, weights, grads, Set{Int}(), true)
end

function forward!(f::Lookup, v::Variable)
  v.value = lookup(f, v[1].value)
end

function lookup{K,V}(f::Lookup{K,V}, x::Vector{K})
  y = alloc_cpu(V, length(f.weights[1]), length(x))
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

function backward!(f::Lookup, v::Variable)
  ∇lookup!(f, v[1].value, v.grad)
end

function ∇lookup!{K,V}(f::Lookup{K,V}, x::Vector{K}, gy::Matrix{V})
  for i = 1:length(x)
    id = f.dict[x[i]]
    f.grads[id] += gy[:, i]
    union!(f.idset, id)
  end
end

function optimize!(opt::Optimizer, f::Lookup)
  for id in f.idset
    update!(opt, f.weights[id], f.grads[id])
  end
  empty!(f.idset)
end
