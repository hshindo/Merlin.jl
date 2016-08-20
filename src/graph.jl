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

function save_hdf5(path::String, name::String, obj)
    h5open(path, "w") do h
        write_hdf5(h, name, obj)
    end
end

function write_hdf5(parent, name::String, obj)
    g = g_create(parent, name)
    attrs(g)["type"] = string(typeof(obj))
    h5 = to_hdf5(obj)
    if typeof(h5) <: Dict
        for (k,v) in h5
            write_hdf5(g, k, v)
        end
    else
        g[name] = h5
    end
end

function to_hdf5(g::Graph)
    dict = Dict()
    for i = 1:length(g)
        dict[i] = to_hdf5(g[i])
    end
    dict
end

function to_hdf5(node::GraphNode)
    dict = Dict()
    for i = 1:length(node.args)
        dict[i] = to_hdf5(node.args[i])
    end
    dict
end

to_hdf5(x::Function) = string(x)
to_hdf5(x::Number) = x
to_hdf5{T<:Number}(x::Array{T}) = x
to_hdf5{T<:Number}(x::Tuple{Vararg{T}}) = T[x...]
to_hdf5(x::Symbol) = string(x)
