export GraphNode, Graph, @graph

type GraphNode
    args::Vector

    GraphNode(args...) = new(Any[args...])
end

type Graph
end

function Graph(top::GraphNode)
    nodes = topsort(top)
    dict = Dict{GraphNode,Expr}()
    for i = length(nodes):-1:1
        node = nodes[i]
        args = map(node.args) do a
            typeof(a) == GraphNode ? dict[a] : a
        end
        dict[node] = Expr(:call, args)
    end
    f(top)
    dict[top]
end

#=
type Graph
    nodes::Vector{Var} # sorted in topological order
    tails::Vector{Vector{Int}}
    names::Dict{Symbol,Int}
end

function Graph(top::GraphNode)
    nodes = topsort(top)
    tails = Vector{Int}[]
    names = Dict{Symbol,Int}()
    dict = ObjectIdDict()
    for i in 1:length(nodes)
        n = nodes[i]
        typeof(n.args[1]) == Symbol && (names[n.args[1]] = i)
        tailids = Int[map(t -> dict[t], n.tails)...]
        push!(tails, tailids)
        dict[n] = i
    end
    Graph(nodes, tails, names)
end

@compat function (g::Graph)(args...)
    outs = Array(Var, length(g.nodes))
    if typeof(args[1]) <: Pair
        for (k,v) in args
            id = g.names[k]
            outs[id] = typeof(v) <: Var ? v : Data(v)
        end
    else
        for i = 1:length(args)
            v = args[i]
            outs[i] = typeof(v) <: Var ? v : Data(v)
        end
    end
    forward!(g, outs)
    outs[end]
end

function forward!(g::Graph, outs::Vector)
    for i = 1:length(g.nodes)
        isdefined(outs, i) && continue
        var, tails = g.nodes[i], g.tails[i]
        if isempty(tails)
            outs[i] = var
        else
            n = length(tails)
            if n == 1
                outs[i] = var(outs[tails[1]])
            elseif n == 2
                outs[i] = var(outs[tails[1]], outs[tails[2]])
            elseif n == 3
                outs[i] = var(outs[tails[1]], outs[tails[2]], outs[tails[3]])
            else
                args = map(id -> outs[id], tails)
                outs[i] = var(args...)
            end
        end
    end
    outs
end
=#

Base.size(v::GraphNode) = GraphNode(size, v)
Base.size(v::GraphNode, dim::Int) = GraphNode(size, v, dim)

macro graph(expr)
    quote
        Graph($(esc(expr)))
    end
end

function hdf5(g::Graph)

end

function to_hdf5(g::Graph)
  d_nodes = Dict()
  for i = 1:length(g.nodes)
    d_nodes[string(i)] = to_hdf5(g.nodes[i])
  end
  d_sym2id = Dict()
  for (k,v) in g.sym2id
    d_sym2id[string(k)] = v
  end
  Dict("Graph" => Dict("nodes" => d_nodes, "sym2id" => d_sym2id))
end

macro graph2(expr)
    isnode(ex) = typeof(ex) == Expr && ex.head == :call && ex.args[1] == :GraphNode
    function conv(ex::Expr)
        for a in ex.args
            typeof(a) == Expr && conv(a)
        end
        ex.head == :call || return
        any(isnode, ex.args) && unshift!(ex.args, :GraphNode)
    end
    conv(expr)
    expr
    #quote
    #    Graph($(esc(expr)))
    #end
end
