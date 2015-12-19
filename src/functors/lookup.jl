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
  Lookup(Dict{K, Int}(), weights, grads, Set{Int}(), false)
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

function apply{K,V}(fun::Lookup{K,V}, input::Vector{K})
  output = Array(V, length(fun.weights[1]), length(input))
  for i = 1:length(input)
    key = input[i]
    def = fun.readonly ? 1 : length(fun.dict) + 1 # TODO: must be fixed
    id = get!(fun.dict, key, def)
    if id > length(fun.weights)
      push!(fun.weights, randn(size(output, 1)))
      push!(fun.grads, zeros(size(output, 1)))
    end
    output[:, i] = fun.weights[id]
  end
  output, gy -> diff(fun, input, gy)
end

function diff{K,V}(fun::Lookup{K,V}, input::Vector{K}, gradout::Matrix{V})
  for i = 1:length(input)
    id = fun.dict[input[i]]
    fun.grads[id] += gradout[:, i]
    union!(fun.idset, id)
  end
  nothing
end

function optimize!(opt::Optimizer, l::Lookup)
  for id in l.idset
    update!(opt, l.weights[id], l.grads[id])
  end
  empty!(l.idset)
end
