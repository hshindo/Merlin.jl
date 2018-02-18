export Graph, Node

mutable struct Node
    f
    args::Tuple
    name::String

    Node(f, args...; name="") = new(f, args, name)
end
Node(; name="") = Node(nothing, name=name)

Base.getindex(x::Node, i::Int) = x.args[i]

struct NodeId
    id::Int
end


struct Graph
    nodes::Vector{Node} # topological order
    inids::Tuple{Vararg{Int}}
    outids::Tuple{Vararg{Int}}
end

function Graph(outs::Node...)
    nodes = topsort(outs...)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))
    dict = Dict{String,Node}()
    for node in nodes
        node.args = map(node.args) do arg
            isa(arg,Node) ? NodeId(node2id[arg]) : arg
        end
        isempty(node.name) || (dict[node.name] = node)
    end

    names = collect(keys(dict))
    sort!(names)
    inids = map(x -> node2id[dict[x]], tuple(names...))
    outids = map(x -> node2id[x], outs)
    Graph(nodes, inids, outids)
end

Base.getindex(g::Graph, i::Int) = g.nodes[i]
Base.getindex(g::Graph, s::String) = g.dict[s]

function (g::Graph)(xs...)
    @assert length(xs) == length(g.inids)
    temps = Array{Any}(length(g.nodes))
    for i = 1:length(xs)
        temps[g.inids[i]] = xs[i]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        if isempty(node.args)
            isassigned(temps,i) || (temps[i] = node.f)
        else
            args = map(node.args) do arg
                isa(arg,NodeId) ? temps[arg.id] : arg
            end
            temps[i] = node.f(args...)
        end
    end
    if length(g.outids) == 1
        temps[g.outids[1]]
    else
        map(id -> temps[id], g.outids)
    end
end
