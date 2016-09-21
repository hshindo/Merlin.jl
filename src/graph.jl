export GraphNode, compile

type GraphNode
    args::Tuple
    name::Symbol

    GraphNode(args...) = new(args, gensym())
end

function tails(n::GraphNode)
    t = GraphNode[]
    for a in n.args
        if typeof(a) == GraphNode
            push!(t, a)
        elseif typeof(a) == Vector{GraphNode}
            append!(t, a)
        end
    end
    t
end

Base.length(n::GraphNode) = length(n.args)
Base.getindex(n::GraphNode, key::Int) = n.args[key]
Base.setindex!(n::GraphNode, value, key::Int) = n.args[key] = value

type Graph
    nodes::Vector{GraphNode} # sorted in bottom-up order
    f
end

function compile(top::GraphNode)
    nodes = topsort(top)
    block = Expr(:block)
    leaf = Expr(:tuple)
    for node in nodes
        if length(node.args) == 0
            push!(leaf.args, node.name)
            continue
        end
        args = map(node.args) do arg
            typeof(arg) == GraphNode ? arg.name : arg
        end
        ex = Expr(:(=), node.name, Expr(:call, args...)) # name = f(args...)
        push!(block.args, ex)
    end
    sort!(leaf.args)
    f = eval(Expr(:->, leaf, block))
    Graph(nodes, f)
end

(g::Graph)(x...) = g.f(x...)

Base.length(g::Graph) = length(g.nodes)
Base.getindex(g::Graph, key::Int) = g.nodes[key]
Base.setindex!(g::Graph, value::GraphNode, key::Int) = g.nodes[key] = value

macro graph2(expr)
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
