export LookupLinear

type LookupLinear
  ws::Vector{Var}
  cache::Dict
end

@compat function(f::LookupLinear)(args::Vector{Var})
  x = args[1]
  y = lookuplinear()
  Var(y, nothing, args)
end

function lookuplinear(cache::Dict{Int,Var}, ws::Vector{Var}, x::Array{Int})
  ys = map(x) do id
    get(cache, id)
  end
  y = reduce(+, ys)
  y
end
