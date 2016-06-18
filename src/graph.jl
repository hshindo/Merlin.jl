export @graph

type GraphNode
  value
  f
  tails::Vector{Int}
end

type Graph
  nodes::Vector{GraphNode} # sorted in bottom-up order
  symtoid::Dict{Symbol,Int}
end

function Graph(top::Var)
  vars = topsort(top)
  symtoid = Dict{Symbol,Int}()
  nodes = Array(GraphNode, length(vars))
  dict = ObjectIdDict()
  for i in 1:length(vars)
    v = vars[i]
    nodes[i] = GraphNode(v.value, v.f, Int[])
    typeof(v.value) == Symbol && (symtoid[v.value] = i)
    dict[v] = i
    for a in v.args
      id = dict[a]
      push!(nodes[i].tails, id)
    end
  end
  Graph(nodes, symtoid)
end

@compat function (g::Graph)(args::Pair{Symbol,Var}...)
  vars = Array(Var, length(g.nodes))
  for (k,v) in args
    id = g.symtoid[k]
    vars[id] = v
  end
  for i = 1:length(g.nodes)
    isdefined(nodes, i) && continue
    n = g.nodes[i]
    if isempty(n.tails)
      vars[i] = n
    else
      args = map(id -> vars[id], n.tails)
      vars[i] = n.f(args)
    end
  end
  vars[end]
end

"""
    Graph(top::Var)

Construct a static network from `var`.

## 👉 Example
Here is an example of three-layer network.
```julia
f = @graph begin
  T = Float32
  x = Var(:x)
  x = Linear(T,10,7)(x)
  x = relu(x)
  x = Linear(7,3)(x)
  x
end
x = Var(rand(Float32,10,5))
y = f(:x=>x)
```
"""
macro graph(src)
  quote
    Graph(eval($src))
  end
end
