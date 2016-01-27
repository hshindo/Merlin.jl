type Lookup{K,V} <: Functor
  keydict::Dict{K,Int}
  values::Vector{Variable}
  unknown::Int
  idset::Set{Int}
end

function Lookup{K,V}(keys::Vector{K}, values::Vector{V})
  all(v -> length(v) == length(values[1]), values) || error("Value length unmatch")
  keydict = Dict{K,Int}()
  sizehint!(keydict, length(keys))
  for i = 1:length(keys)
    keydict[keys[i]] = i
  end
  Lookup(keydict, values, Set{Int}())
end

function Lookup{K,V}(keys::Vector{K}, ::Type{V}, vlength::Int)
  values = convert(Vector{V}, randn(vlength))
  Lookup(keys, values)
end

forward!(f::Lookup, v::Variable) = v.value = lookup(f, v[1].value)

function lookup{K,V}(f::Lookup{K,V}, x::Vector{K})
  y = Array(V, length(f.embeds[1]), length(x))
  for i = 1:length(x)
    id = get(f.iddict, x[i], 1)
    y[:, i] = f.values[id]
  end
  y
end
