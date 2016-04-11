export Sequence

type Sequence <: Functor
  funs::Vector{Functor}
end
Sequence(funs::Functor...) = Sequence([funs...])

@compat function (seq::Sequence)(arg::Variable)
  y = arg
  for f in seq.funs
    y = f(y)
  end
  y
end

function update!(opt::Optimizer, seq::Sequence)
  for f in seq.funs
    applicable(update!, opt, f) && update!(opt, f)
  end
end
