export Graph
export @graph

type Graph
  nodes::Vector # sorted in bottom-up order
  names::Dict{Symbol,Int}
end

function Graph(top::Layer)
  nodes = topsort(top)
  names = Dict{Symbol,Int}()
  dict = ObjectIdDict()
  for i in 1:length(nodes)
    n = nodes[i]
    typeof(n) <: Data && (names[n.name] = i)
    dict[n] = i
  end
  Graph(nodes, names)
end

@compat function (g::Graph)(args::Pair...)
  for (k,v) in args
    g[k].data = typeof(v) <: Data ? v.data : v
  end
  for n in g.nodes
    forward!(n)
  end
  g.nodes[end]
end

Base.getindex(g::Graph, key::Int) = g.nodes[key]
Base.getindex(g::Graph, key::Symbol) = g[g.names[key]]

macro graph(expr)
  quote
    Graph(eval($(esc(expr))))
  end
end
