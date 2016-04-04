type Sequence <: Functor
  funs::Vector{Functor}
end

function call(seq::Sequence, v::Variable)
  for f in seq.funs
    v = f(v)
  end
  v
end
