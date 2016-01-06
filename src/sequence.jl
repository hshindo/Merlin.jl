type Sequence <: Functor
  fs::Vector
  x
  y
end

Sequence(fs::Vector) = Sequence(fs, nothing, nothing)
Sequence(fs...) = Sequence([fs...], nothing, nothing)

clone(s::Sequence) = Sequence(map(clone, s.fs))

function forward!(s::Sequence)
  x = s.x
  for f in s.fs
    x = f(x)
  end
  s.y = x
end

function optimize!(opt::Optimizer, seq::Sequence)
  for fun in seq.funs
    applicable(optimize!, opt, fun) && optimize!(opt, fun)
  end
end
