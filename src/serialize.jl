function save(g::Graph)
  for n in g.nodes
    n
  end
end

function save_f(f, filename)
  h5open(filename, "w") do f
    write(f, "/", A)
  end
end

function todict{T}(x::T)
  d = Dict()
  names = fieldnames(T)
  for i = 1:nfields(T)
    f = getfield(x, i)
    d[string(names[i])] = f
  end
  d
end
