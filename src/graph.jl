export @graph, compile

type GraphNode
    args::Vector

    GraphNode(args...) = new(Any[args...])
end

Base.length(n::GraphNode) = length(n.args)
Base.getindex(n::GraphNode, key::Int) = n.args[key]
Base.setindex!(n::GraphNode, value, key::Int) = n.args[key] = value

type Graph
    nodes::Vector{GraphNode} # sorted in bottom-up order
end

Graph(top::GraphNode) = Graph(topsort(top))

"""
    compile(g::Graph, args::Symbol...)

Compile a computational graph and generate an anonymous function.
"""
function compile(g::Graph, args::Symbol...)
    dict = ObjectIdDict()
    for node in g.nodes
        exprs = map(node.args) do n
            typeof(n) == GraphNode ? dict[n] : n
        end
        dict[node] = Expr(:call, exprs...)
    end
    expr = Expr(:->, Expr(:tuple, args...), dict[g.nodes[end]]) # create anonymous function
    eval(expr)
end

Base.length(g::Graph) = length(g.nodes)
Base.getindex(g::Graph, key::Int) = g.nodes[key]
Base.setindex!(g::Graph, value::GraphNode, key::Int) = g.nodes[key] = value

macro graph(expr)
    bottomup(expr) do ex
        if ex.head == :call
            unshift!(ex.args, :(Merlin.GraphNode))
        end
    end
    quote
        Graph($(esc(expr)))
    end
end

function bottomup{T}(f, node::T)
    for arg in node.args
        typeof(arg) == T && bottomup(f, arg)
    end
    f(node)
end

function to_hdf5(g::Graph)
    dict = Dict()
    argdict = ObjectIdDict()
    for i = 1:length(g)
        d = Dict()
        dict[i] = d
        for j = 1:length(g[i])
            n = g[i][j]
            key = "$(j)::$(typeof(n))"
            if typeof(n) == GraphNode
                d[key] = argdict[n]
            else
                d[key] = to_hdf5(n)
            end
        end
        argdict[g[i]] = i
    end
    dict
end

function from_hdf5(::Type{Graph}, dict::Dict)
    nodes = GraphNode[]
    nodedict = ObjectIdDict()
    for (nodeid,nodedict) in dict
        args = []
        for (k,v) in nodedict
            exprs = parse(k).args
            id, T = exprs[1], eval(exprs[2])
            args[id] = from_hdf5(T, v)
        end
        GraphNode(args...)
    end
    Graph(nodes, nothing)
end
