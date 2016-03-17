type Cache <: Functor
  f::Functor
  dict::Dict
end

function forward!(f::Cache, v::Variable)
  if haskey(f.dict, v.value)
    y = dict[v.value]
  else
    y = f(v)
    dict[v.value] = y
  end
  y
end

type Custom <: Functor
  fun::Function
end

forward!(f::Custom, v::Variable) = f.fun(v)
