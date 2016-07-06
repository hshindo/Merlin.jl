export GraphNode, @graph

type GraphNode <: Layer
  f
  args::Vector
  tails::Vector{Int}
end

GraphNode(f, args...) = GraphNode(f, Any[args...], Int[])

tails(node::GraphNode) = filter(a -> typeof(a) <: Layer, node.args)

type Graph
  nodes::Vector # sorted in bottom-up order
  sym2id::Dict{Symbol,Int}
end

function Graph(top::GraphNode)
  nodes = topsort(top)
  sym2id = Dict{Symbol,Int}()
  dict = ObjectIdDict()
  for i in 1:length(nodes)
    n = nodes[i]
    n.tails = map(a -> get(dict, a, 0), n.args)
    isempty(n.args) && (sym2id[n.f] = i)
    dict[n] = i
  end
  Graph(nodes, sym2id)
end

@compat function (g::Graph)(args::Pair...)
  layers = Array(Layer, length(g.nodes))
  for (k,v) in args
    id = g.sym2id[k]
    layers[id] = typeof(v) <: Layer ? v : Data(v)
  end
  for i = 1:length(g.nodes)
    isdefined(layers, i) && continue
    n = g.nodes[i]
    args = Array(Any, length(n.args))
    for k = 1:length(args)
      id = n.tails[k]
      args[k] = id < 0 ? n.args[k] : layers[id]
    end
    layers[i] = n.f(args...)
  end
  layers[end]
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
