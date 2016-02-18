type Lookup <: Functor
  weights::Vector{Variable}
  idset::Set{Int}
end

function Lookup{T}(::Type{T}, size1::Int, size2::Int)
  weights = Variable[]
  for i = 1:size2
    a = AFArray(convert(Vector{T},randn(size1)))
    push!(weights, Variable(a))
  end
  Lookup(weights, Set{Int}())
end

function forward!(f::Lookup, v::Variable)
  ids = v[1].value
  xs = map(id -> f.weights[id].value, ids)
  v.value = cat(xs, 2)
end

function backward!(f::Lookup, v::Variable)
  return nothing
  ids = v[1].value
  for i = 1:length(x)
    id = ids[i]
    #lookup(v.grad, )
    addgrad!(f.weights[id], gy[:, i])
    union!(f.idset, id)
  end
end

function optimize2!(opt::Optimizer, f::Lookup)
  for id in f.idset
    p = f.params[id]
    update!(opt, p.value, p.grad)
  end
  empty!(f.idset)
end
