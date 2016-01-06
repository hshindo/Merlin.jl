type Map <: Functor
  fs::Vector{Functor}
  x::Vector
  y::Vector
end

function forward!(f::Map)


  fs, x, y = f.fs, f.x, f.y
  while length(f.fs) < length(f.x)
    push!(f.fs, clone(f.fs[1]))
  end
  f.y = Array(typeof(), length(f.x))
  for i = 1:length(x)
    ys[i] = fs[i](x[i])
  end
end
