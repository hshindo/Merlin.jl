export Graph, @graph

type Graph
  nodes::Vector{Var} # sorted in bottom-up order
  sym2id::Dict{Symbol,Int}
end

function Graph(top::Var, name=:g)
  vars = topsort(top)
  sym2id = Dict{Symbol,Int}()
  nodes = Array(Var, length(vars))
  dict = ObjectIdDict()
  for i in 1:length(vars)
    v = vars[i]
    ids = map(a -> dict[a], v.args)
    nodes[i] = Var(v.value, v.f, ids, v.grad)
    typeof(v.value) == Symbol && v.value != Symbol() && (sym2id[v.value] = i)
    dict[v] = i
  end
  Graph(nodes, sym2id)
end

@compat function (g::Graph)(args::Pair{Symbol,Var}...)
  #settype!(typeof(args[1][2]), g.nodes)

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
