type Lookup <: Functor
  weight::Variable
  idset::Set{Int}
end

function Lookup{T}(::Type{T}, xlength::Int, ylength::Int)
  w = randn(xlength, ylength)
  w = convert(Matrix{T}, w)
  v = Variable(w, true)
  Lookup(v, Set{Int}())
end

function forward!(f::Lookup, v::Variable)
  indices = v[1].value
  v.value = lookup(f.weight.value, indices, 2)
end

function optimize!(opt::Optimizer, f::Lookup)
  for id in f.idset
    p = f.params[id]
    update!(opt, p.value, p.grad)
  end
  empty!(f.idset)
end
