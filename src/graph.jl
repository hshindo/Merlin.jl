export @graph

type GraphNode
    args::Vector

    GraphNode(args...) = new(Any[args...])
end

Base.length(n::GraphNode) = length(n.args)
Base.getindex(n::GraphNode, key::Int) = n.args[key]
Base.setindex!(n::GraphNode, value, key::Int) = n.args[key] = value

type Graph
    nodes::Vector{GraphNode} # sorted in bottom-up order
    f
end

function Graph(nodes::Vector{GraphNode}, args::Vector{Symbol})
    dict = ObjectIdDict()
    for node in nodes
        exprs = map(node.args) do n
            typeof(n) == GraphNode ? dict[n] : n
        end
        dict[node] = Expr(:call, exprs...)
    end
    expr = Expr(:->, Expr(:tuple, args...), dict[nodes[end]]) # create anonymous function
    f = eval(expr)
    Graph(nodes, f)
end

(g::Graph)(x...) = g.f(x...)

Base.length(g::Graph) = length(g.nodes)
Base.getindex(g::Graph, key::Int) = g.nodes[key]
Base.setindex!(g::Graph, value::GraphNode, key::Int) = g.nodes[key] = value

macro graph(expr)
    local dict = ObjectIdDict()
    local syms = Symbol[]
    bottomup(expr) do ex
        if ex.head == :call
            unshift!(ex.args, :(Merlin.GraphNode))
        end
        if ex.head == :quote && length(ex.args) == 1
            local s = ex.args[1]
            if typeof(s) == Symbol && !haskey(dict, s)
                dict[s] = s
                push!(syms, s)
            end
        end
    end
    quote
        local top = $(esc(expr))
        Graph(topsort(top), $syms)
    end
end

function bottomup{T}(f, node::T)
    for arg in node.args
        typeof(arg) == T && bottomup(f, arg)
    end
    f(node)
end

function h5convert(x::Graph)
    dict = h5dict(Graph)
    argdict = ObjectIdDict()
    for i = 1:length(x)
        d = Dict{String,Any}()
        dict[string(i)] = d
        for j = 1:length(x[i])
            n = x[i][j]
            if typeof(n) == GraphNode
                d[string(j)] = Dict("#NODE"=>argdict[n])
            else
                d[string(j)] = h5convert(n)
            end
        end
        argdict[x[i]] = i
    end
    dict
end

function h5load!(::Type{Graph}, data::Dict)
    nodes = GraphNode[]
    for (k,v) in data
        args = h5load!(Vector{Any}, v)
        id = parse(Int, k)
        while id > length(nodes)
            push!(nodes, GraphNode())
        end
        nodes[id] = GraphNode(args...)
    end
    for node in nodes
        for i = 1:length(node)
            typeof(node[i]) <: Dict || continue
            haskey(node[i], "#NODE") || continue
            id = node[i]["#NODE"]
            node[i] = nodes[id]
        end
    end
    Graph(nodes)
end
