export Graph, @graph

type GraphNode
  f
  tails::Vector{Int}
end

type Graph
  nodes::Vector # sorted in bottom-up order
  sym2id::Dict{Symbol,Int}
end

function Graph(top::Layer)
  layers = topsort(top)
  nodes = Array(GraphNode, length(layers))
  sym2id = Dict{Symbol,Int}()
  dict = ObjectIdDict()

  for i in 1:length(layers)
    l = layers[i]
    ids = map(t -> dict[t], tails(l))
    nodes[i] = GraphNode(l, ids)
    typeof(l.y) == Symbol && v.value != Symbol() && (sym2id[v.value] = i)
    #nodes[i] = Var(v.value, v.f, ids, v.grad)
    #typeof(v.value) == Symbol && v.value != Symbol() && (sym2id[v.value] = i)
    dict[l] = i
  end
  Graph(nodes, sym2id)
end

@compat function (g::Graph)(args::Pair...)
  vars = Array(Var, length(g.nodes))
  for (k,v) in args
    id = g.sym2id[k]
    vars[id] = v
  end
  for i = 1:length(g.nodes)
    isdefined(vars, i) && continue
    n = g.nodes[i]
    if isempty(n.args)
      vars[i] = n
    else
      args = map(id -> vars[id], n.args)
      vars[i] = typeof(args) <: Tuple ? n.f(args...) : n.f(args)
    end
  end
  vars[end]
end

"""
    @graph(top::Var)

Construct a static network from `var`.
"""
macro graph(src)
  quote
    Graph(eval($(esc(src))))
  end
end
