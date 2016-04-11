export Graph
export Sequence

type Graph <: Functor
  vars::Vector{Variable} # sorted in bottom-up order
  inids::Vector{Int}
end

function Graph(var::Variable)
  sorted = topsort(var)
  inids = Int[]
  for i in 1:length(sorted)
    v = sorted[i]
    length(v.args) == 0 && v.value == nothing && push!(inids, i)
    v.value = i
  end

  # compile
  for v in sorted

  end

  Graph(sorted, inids)
end

function Sequence(funs::Functor...)
  v = Variable()
  for f in funs
    v = f(v)
  end
  Graph(v)
end

Base.getindex(g::Graph, key) = g.vars[key]

@compat function (g::Graph)(args...)
  @assert (length(g.inids) == length(args))
  vars = Array(Variable, length(g.vars))
  for i in 1:length(args)
    vars[g.inids[i]] = args[i]
  end
  for i = 1:length(vars)
    isdefined(vars, i) && continue
    v = g[i]
    if length(v.args) == 1
      vars[i] = v.f(vars[v[1].value])
    else
      args = map(a -> vars[a.value], v.args)
      vars[i] = v.f(args)
    end
    #args = map(a -> vars[a.value], v.args)
    #vars[i] = v.f(args)
  end
  vars[end]
end

function update!(opt::Optimizer, g::Graph)
  for v in g.vars
    applicable(update!, opt, v.f) && update!(opt, v.f)
  end
end
