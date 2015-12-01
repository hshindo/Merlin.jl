type Sequence <: Functor
  funs::Vector{Functor}
end

Sequence(funs::Functor...) = Sequence([funs...])

function apply(seq::Sequence, vars::Variable...)
  var = apply(seq.funs[1], vars...)
  for i = 2:length(seq.funs)
    var = apply(seq.funs[i], var)
  end
  var
end
