type Lookup <: Layer
  weights::Vector{Variable}
  idset::Set{Int}
  args
  out

  function Lookup(weights, idset)
    l = Lookup(weights, idset, nothing, nothing)
    finalizer(l, free)
    l
  end
end

function Lookup{T}(::Type{T}, size1::Int, size2::Int)
  weights = Variable[]
  for i = 1:size2
    a = AFArray(convert(Vector{T},randn(size1)))
    push!(weights, Variable(a))
  end
  Lookup(weights, Set{Int}())
end

function forward!(l::Lookup)
  ids = l.args[1].value
  xs = map(id -> f.weights[id].value, ids)
  l.out.value = cat(xs, 2)
end

function free(l::Lookup)
  for id in l.idset
    release(l.weights[id].value)
  end
end
