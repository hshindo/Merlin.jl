export @graph

type ExprNode
  f
  tails::Vector{ExprNode}
end

#Base.invoke(f, tails::NTuple{1,ExprNode}) = f(tails[1])
#Base.invoke(f, tails::NTuple{2,ExprNode}) = f(tails[1])

ExprNode(f, args...) = ExprNode(f, Any[args...])

tails(v::ExprVar) = filter(a -> typeof(a) <: Var, v.args)

function aaa(v::ExprVar, vars::Vector)
  args = Array(Any, length(v.args))
  for i = 1:length(args)
    id = v.argids[i]
    args[i] = id == 0 ? v.args[i] : vars[id]
  end
  v.f(args...)
end

type ExprGraph
  nodes::Vector{ExprNode} # sorted in bottom-up order
  names::Dict{Symbol,Int}
end

function ExprGraph(top::ExprNode)
  nodes = topsort(top)
  names = Dict{Symbol,Int}()
  dict = ObjectIdDict()
  for i in 1:length(nodes)
    n = nodes[i]


    n.tails = map(a -> get(dict, a, 0), n.args)
    isempty(n.tails) && (names[n.f] = i)
    dict[n] = i
  end
  Graph(nodes, names)
end

@compat function (g::Graph)(args::Pair...)
  vars = Array(Var, length(g.nodes))
  for (k,v) in args
    id = g.names[k]
    vars[id] = typeof(v) <: Var ? v : Data(v)
  end
  for i = 1:length(g.nodes)
    isdefined(vars, i) && continue
    n = g.nodes[i]
    args = Array(Any, length(n.args))
    for k = 1:length(args)
      id = n.tails[k]
      args[k] = id < 0 ? n.args[k] : vars[id]
    end
    vars[i] = n.f(args...)
  end
  vars[end]
end

#=
function Graph(top::GraphNode)
  nodes = topsort(top)
  names = Dict{Symbol,Int}()
  dict = ObjectIdDict()
  for i in 1:length(nodes)
    n = nodes[i]
    typeof(n) <: Data && n.name != Symbol() && (names[n.name] = i)
    dict[n] = i
  end
  Graph(nodes, names)
end
=#

#=
@compat function (g::Graph)(args::Pair...)
  outputs = Array(Layer, length(g.nodes))
  for (k,v) in args
    id = g.names[k]
    outputs[id] = typeof(v) <: Data ? v : Data(v)
  end
  for i = 1:length(g.nodes)
    n = g.nodes[i]
    xs = map(id -> outputs[id], g.tails[i])
    typeof(n)(xs...)
  end
  outputs[end]
end
=#

macro graph(expr)
  quote
    Graph(eval($(esc(expr))))
  end
end

function topsort(top)
  sorted = []
  dict = ObjectIdDict()
  function visit(v)
    haskey(dict, v) && return
    dict[v] = v
    for t in tails(v)
      visit(t)
    end
    push!(sorted, v)
  end
  visit(top)
  sorted
end
