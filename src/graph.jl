export @graph

type GraphNode <: AbstractNode
    args::Vector

    GraphNode(args...) = new(Any[args...])
end

type Graph
    nodes::Vector{GraphNode} # sorted in bottom-up order
    f
end

function Graph(top::GraphNode, syms::Tuple{Vararg{Symbol}})
    nodes = topsort(top)
    dict = ObjectIdDict()
    for node in nodes
        args = map(node.args) do n
            typeof(n) == GraphNode ? dict[n] : n
        end
        dict[node] = Expr(:call, args...)
    end
    expr = Expr(:->, Expr(:tuple, syms...), dict[top]) # create anonymous function
    f = eval(expr)
    Graph(nodes, f)
end

Base.length(g::Graph) = length(g.nodes)
Base.getindex(g::Graph, key::Int) = g.nodes[key]
Base.setindex!(g::Graph, value::GraphNode, key::Int) = g.nodes[key] = value

(g::Graph)(xs...) = g.f(xs...)

macro graph(args, expr)
    bottomup(expr) do ex
        if ex.head == :call
            unshift!(ex.args, :(Merlin.GraphNode))
        end
    end
    quote
        Graph($(esc(expr)), $args)
    end
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
