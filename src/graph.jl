export Graph

"""
# Graph

`Graph` is a container of `Functor`s.
The following is an example of Gated Recurrent Unit (GRU).

### 👉 Example
```julia
# parameters
Ws = [Variable(rand(T,xsize,xsize)) for i=1:3]
Us = [Variable(rand(T,xsize,xsize)) for i=1:3]
# input
x = Variable()
h = Variable()

r = sigmoid(Ws[1]*x + Us[1]*h)
z = sigmoid(Ws[2]*x + Us[2]*h)
h_ = tanh(Ws[3]*x + Us[3]*(r.*h))
h_next = (1 - z) .* h + z .* h_
Graph(h_next)
```
"""
type Graph <: Functor
  nodes::Vector{Variable} # sorted in bottom-up order
  tail_ids::Vector{Vector{Int}}
  data_ids::Vector{Int}
end

function Graph(funs::Functor...)
  v = Variable()
  for f in funs
    v = f(v)
  end
  Graph(v)
end

function Graph(top::Variable)
  nodes = topsort(top)
  data_ids = Int[]
  for i in 1:length(nodes)
    v = nodes[i]
    length(v.args) == 0 && v.value == nothing && push!(data_ids, i)
  end

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
end

Base.getindex(g::Graph, key) = g.nodes[key]

@compat function (g::Graph)(args::Tuple)
  @assert (length(g.data_ids) == length(args))

  nodes = Array(Variable, length(g.nodes))
  for i in 1:length(args)
    a = args[i]
    v = typeof(a) == Variable ? a : Variable(a, nothing)
    nodes[g.data_ids[i]] = v
  end

  for i = 1:length(nodes)
    isdefined(nodes, i) && continue
    v = g[i]
    tails = g.tail_ids[i]
    if length(tails) == 0 # param
      nodes[i] = v
    elseif length(tails) == 1
      nodes[i] = v.f(nodes[tails[1]])
    else
      inputs = map(id -> nodes[id], tails)
      nodes[i] = v.f(inputs)
    end
  end
  nodes[end]
end

@compat (g::Graph)(args...) = g(args)
