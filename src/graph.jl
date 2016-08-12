export compile, @graph

type GraphNode <: AbstractNode
    args::Vector

    GraphNode(args...) = new(Any[args...])
end

type Graph
    top::GraphNode
    f
end

function Graph()
end

@compat (g::Graph)(xs...) = g.f(xs...)

function compile(top::GraphNode, syms::Tuple{Vararg{Symbol}})
    nodes = topsort(top)
    dict = ObjectIdDict()
    for node in nodes
        args = map(node.args) do n
            typeof(n) == GraphNode ? dict[n] : n
        end
        dict[node] = Expr(:call, args...)
    end
    expr = Expr(:->, Expr(:tuple, syms...), dict[top]) # create anonymous function
    eval(expr)
end

macro graph(args, expr)
    bottomup(expr) do ex
        if ex.head == :call
            unshift!(ex.args, :(Merlin.GraphNode))
        end
    end
    quote
        local top = $(esc(expr))
        local f = compile(top, $args)
        Graph(top, f)
    end
end

to_hdf5(x::Function) = string(x)
to_hdf5(x::Number) = x
to_hdf5{T}(data::Tuple{Vararg{T}}) = T[data...]
to_hdf5(s::Symbol) = string(s)

function to_hdf5(g::Graph)
    dict = Dict()
    nodes = topsort(g.top)
    nodedict = ObjectIdDict()
    for i = 1:length(nodes)
        n = nodes[i]
        d = Dict()
        for j = 1:length(n.args)
            a = n.args[j]
            h5 = typeof(a) == GraphNode ? nodedict[a] : to_hdf5(a)
            d[j] = h5
        end
        dict[i] = Dict("GraphNode" => d)
        nodedict[n] = i
    end
    Dict("Graph" => dict)
end
