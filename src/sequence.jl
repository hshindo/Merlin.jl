export Sequence

type Sequence <: Functor
  funs::Vector{Functor}
  cache::Dict
end
Sequence(funs::Functor...) = Sequence([funs...], Dict())

@compat function (seq::Sequence)(arg::Variable)
  v = arg
  for f in seq.funs
    v = f(v)
  end
  v
end
@compat (f::Sequence)(arg::Data) = f(Variable(arg,nothing))

function update!(opt::Optimizer, seq::Sequence)
  for f in seq.funs
    applicable(update!, opt, f) && update!(opt, f)
  end
end
