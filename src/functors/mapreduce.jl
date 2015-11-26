type MapReduce <: Functor
  mapper::Functor
  reducer::Functor
end

function apply(fun::MapReduce, vars::Vector{Variable})
  mapped = map(x -> apply(fun.mapper, x), var.data)
  reduced = apply(fun.reducer, mapped)
  Variable(reduced.data, (mapped, reduced))
end

function diff()

end
