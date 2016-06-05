export Graph

"""
    Graph

Construct a static network.
"""
type Graph
  nodes::Vector{Var} # sorted in bottom-up order
  vars::Dict{Symbol,Var}
end

function Graph(top::Var)
  nodes = topsort(top)
  
end

@compat function (g::Graph)(args::Tuple{Symbol,Var}...)
  for (key,val) in args
    vars[key].value = val
  end
  nodes = map(n -> n.f(), g.nodes)
  nodes[end]
end

type Graph
  nodes::Vector{Var} # sorted in bottom-up order
  tail_ids::Vector{Vector{Int}}
  data_ids::Vector{Int}
end

function Graph(top::Var)
  nodes = topsort(top)
  data_ids = Int[]
  for i in 1:length(nodes)
    v = nodes[i]
    isempty(v.args) && v.value == nothing && push!(data_ids, i)
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

#=
type Graph
  nodes::Vector{Var} # sorted in bottom-up order
  tail_ids::Vector{Vector{Int}}
  data_ids::Vector{Int}
end

function Graph(top::Var)
  nodes = topsort(top)
  data_ids = Int[]
  for i in 1:length(nodes)
    v = nodes[i]
    isempty(v.args) && v.value == nothing && push!(data_ids, i)
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
