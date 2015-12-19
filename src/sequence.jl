type Sequence
  funs::Vector
end

Sequence(funs...) = Sequence([funs...])

function optimize!(opt::Optimizer, seq::Sequence)
  for fun in seq.funs
    applicable(optimize!, opt, fun) && optimize!(opt, fun)
  end
end
