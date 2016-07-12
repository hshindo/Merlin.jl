export Graph
export @graph

type Graph
  nodes::Vector{Var} # sorted in topological order
  tails::Vector{Vector{Int}}
  names::Dict{Symbol,Int}
end

function Graph(top::Var)
  nodes = topsort(top)
  tails = Vector{Int}[]
  names = Dict{Symbol,Int}()
  dict = ObjectIdDict()
  for i in 1:length(nodes)
    n = nodes[i]
    typeof(n.data) <: Symbol && (names[n.data] = i)
    ids = Int[map(t -> dict[t], n.tails)...]
    push!(tails, ids)
    dict[n] = i
  end
  Graph(nodes, tails, names)
end

@compat function (g::Graph)(args...)
  outs = Array(Any, length(g.nodes))
  if typeof(args[1]) <: Pair
    for (k,v) in args
      id = g.names[k]
      outs[id] = typeof(v) <: Var ? v : Data(v)
    end
  else
    for i = 1:length(args)
      v = args[i]
      outs[i] = typeof(v) <: Var ? v : Data(v)
    end
  end
  forward!(g, outs)
  outs[end]
end

function forward!(g::Graph, outs::Vector)
  for i = 1:length(g.nodes)
    isdefined(outs, i) && continue
    n = g.nodes[i]
    tails = g.tails[i]
    if isempty(tails)
      outs[i] = n
    else
      args = map(id -> outs[id], tails)
      outs[i] = forward(n, args...)
    end
  end
end

macro graph(expr)
  quote
    Graph(eval($(esc(expr))))
  end
end

function topsort(top::Var)
  sorted = Var[]
  dict = ObjectIdDict()
  function visit(v)
    haskey(dict, v) && return
    dict[v] = v
    for t in v.tails
      visit(t)
    end
    push!(sorted, v)
  end
  visit(top)
  sorted
end
