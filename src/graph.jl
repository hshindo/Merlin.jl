export Graph, @graph

type Graph
  nodes::Vector{Var} # sorted in bottom-up order
  tailids::Vector{Vector{Int}}
  #data_ids::Vector{Int}
  iddict::Dict{Symbol,Int}
end

function Graph(top::Var)
  nodes = topsort(top)
  iddict = Dict{Symbol,Int}()
  #data_ids = Int[]
  for i in 1:length(nodes)
    v = nodes[i]
    typeof(v.value) == Symbol && (iddict[v.value] = i)
    #isempty(v.args) && v.value == nothing && push!(data_ids, i)
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
#=
  dict = ObjectIdDict()
  for i = 1:length(nodes)
    dict[nodes[i]] = i
  end

  tail_ids = [Int[] for i=1:length(nodes)]
  for i = 1:length(nodes)
    for a in nodes[i].args
      push!(tail_ids[i], dict[a])
    end
  end
  Graph(nodes, tail_ids, data_ids)
=#
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
    #elseif length(tailids) == 1
    #  nodes[i] = forward(v.f, [nodes[tails[1]]])
    #else
    #  inputs = map(id -> nodes[id], tails)
    #  nodes[i] = forward(v.f, inputs)
    end
  end
  nodes[end]
end

#=
@compat function (g::Graph)(args::Vector{Var})
  @assert length(g.data_ids) == length(args)

  nodes = Array(Var, length(g.nodes))
  for i in 1:length(args)
    nodes[g.data_ids[i]] = args[i]
  end

  for i = 1:length(nodes)
    isdefined(nodes, i) && continue
    v = g[i]
    tails = g.tail_ids[i]
    if isempty(tails) # param
      nodes[i] = v
    elseif length(tails) == 1
      nodes[i] = forward(v.f, [nodes[tails[1]]])
    else
      inputs = map(id -> nodes[id], tails)
      nodes[i] = forward(v.f, inputs)
    end
  end
  nodes[end]
end
@compat (g::Graph)(args::Var...) = g([args...])
=#

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
