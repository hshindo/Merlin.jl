export Graph

type Graph <: Functor
  vars::Vector{Variable} # sorted in bottom-up order
  tailids::Vector{Vector{Int}}
  inids::Vector{Int}
end

function Graph(funs::Functor...)
  v = Variable()
  for f in funs
    v = f(v)
  end
  Graph(v)
end

function Graph(var::Variable)
  sorted = topsort(var)
  inids = Int[]
  for i in 1:length(sorted)
    v = sorted[i]
    length(v.args) == 0 && v.value == nothing && push!(inids, i)
    #v.value = i
  end
  for v in sorted
    #compile!(GEMM!, v) && continue
    #compile!(AXPY!, v) && continue
  end

  dict = ObjectIdDict()
  for i = 1:length(sorted)
    dict[sorted[i]] = i
  end
  tailids = [Int[] for i=1:length(sorted)]
  for i = 1:length(sorted)
    for a in sorted[i].args
      push!(tailids[i], dict[a])
    end
  end

  Graph(sorted, tailids, inids)
end

Base.getindex(g::Graph, key) = g.vars[key]

@compat function (g::Graph)(args::Tuple)
  @assert (length(g.inids) == length(args))
  vars = Array(Variable, length(g.vars))
  for i in 1:length(args)
    a = args[i]
    v = typeof(a) == Variable ? a : Variable(a, nothing)
    vars[g.inids[i]] = v
  end

  for i = 1:length(vars)
    isdefined(vars, i) && continue
    v = g[i]
    tails = g.tailids[i]
    if length(tails) == 0 # param
      vars[i] = v
    elseif length(tails) == 1
      vars[i] = v.f(vars[tails[1]])
    else
      args = map(id -> vars[id], tails)
      vars[i] = v.f(args)
    end
  end
  vars[end]
end

@compat (g::Graph)(args...) = g(args)

function update!(opt::Optimizer, g::Graph)
  for v in g.vars
    applicable(update!, opt, v.f) && update!(opt, v.f)
  end
end
