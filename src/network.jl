export Network

"""
## Network

`Network` is a container of `Functor`s.

### 👉 Example
```julia
```
"""
type Network <: Functor
  nodes::Vector{Var} # sorted in bottom-up order
  tail_ids::Vector{Vector{Int}}
  data_ids::Vector{Int}
end

function Network(funs::Functor...)
  v = Var()
  for f in funs
    v = f(v)
  end
  Network(v)
end

function Network(top::Var)
  nodes = topsort(top)
  data_ids = Int[]
  for i in 1:length(nodes)
    v = nodes[i]
    length(v.args) == 0 && v.val == nothing && push!(data_ids, i)
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
  Network(nodes, tail_ids, data_ids)
end

Base.getindex(f::Network, key) = f.nodes[key]

@compat function (f::Network)(args::Vector{Var})
  @assert (length(f.data_ids) == length(args))

  nodes = Array(Var, length(f.nodes))
  for i in 1:length(args)
    nodes[f.data_ids[i]] = args[i]
  end

  for i = 1:length(nodes)
    isdefined(nodes, i) && continue
    v = f[i]
    tails = f.tail_ids[i]
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
@compat (f::Network)(args::Var...) = f([args...])
