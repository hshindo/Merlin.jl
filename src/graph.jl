export Graph, @graph

type Graph
  nodes::Vector{Var} # sorted in bottom-up order
  tailids::Vector{Vector{Int}}
  iddict::Dict{Symbol,Int}
end

function Graph(top::Var)
  nodes = topsort(top)
  iddict = Dict{Symbol,Int}()
  for i in 1:length(nodes)
    v = nodes[i]
    typeof(v.value) == Symbol && (iddict[v.value] = i)
  end

  dict = ObjectIdDict()
  for i = 1:length(nodes)
    dict[nodes[i]] = i
  end
  tailids = [Int[] for i=1:length(nodes)]
  for i = 1:length(nodes)
    for a in nodes[i].args
      push!(tailids[i], dict[a])
    end
  end
  Graph(nodes, tailids, iddict)
end

Base.getindex(g::Graph, key) = g.nodes[key]

@compat function (g::Graph)(args::Pair{Symbol,Var}...)
  nodes = Array(Var, length(g.nodes))
  for (k,v) in args
    id = g.iddict[k]
    nodes[id] = v
  end
  for i = 1:length(g.nodes)
    isdefined(nodes, i) && continue
    n = g[i]
    tailids = g.tailids[i]
    if isempty(tailids) # param
      nodes[i] = n
    else
      xs = map(id -> nodes[id], tailids)
      nodes[i] = n.f(xs)
    end
  end
  nodes[end]
end

"""
    Graph(top::Var)

Construct a static network from `var`.

`Var()` generates a place-holder of input.
When a function is applied to `Var()`, the omputation is lazily evaluated.

### 👉 Example
Here is an example of constructing a three-layer network.
```julia
f = @graph begin
  T = Float32
  x = Var()
  x = Linear(T,10,7)(x)
  x = relu(x)
  x = Linear(7,3)(x)
  x
end
x = Var(rand(Float32,10,5))
y = f(x)
```
"""
macro graph(src)
  quote
    Graph(eval($src))
  end
end
