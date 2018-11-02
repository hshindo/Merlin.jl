export Graph, Node
export graphs

struct Node
    f
    args::Tuple
    name::Symbol
end
Node(f, args) = Node(f, args, Symbol())
Node(; name::Symbol) = Node(nothing, (), name)

Base.getindex(x::Node, i::Int) = x.args[i]

struct NodeId
    id::Int
end

struct Graph
    nodes::Vector{Node} # topological order
    inputids::Tuple
    outputids::Tuple
end

function Graph(outs::Node...)
    nodes = topsort(outs...)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))
    nodes = map(nodes) do n
        args = map(n.args) do arg
            isa(arg,Node) ? NodeId(node2id[arg]) : arg
        end
        Node(n.f, args, n.name)
    end
    name2id = Dict{Symbol,Int}()
    for i = 1:length(nodes)
        n = nodes[i]
        isempty(n.args) && (name2id[n.name] = i)
    end
    inputids = tuple(name2id...)
    outputids = map(x -> node2id[x], outs)
    Graph(nodes, inputids, outputids)
end

Base.getindex(g::Graph, i::Int) = g.nodes[i]
Base.getindex(g::Graph, s::String) = g.dict[s]
graphs(g::Graph) = (g,)

function parameters(g::Graph)
    params = Var[]
    for n in g.nodes
        for arg in n.args
            isparam(arg) && push!(params,arg)
        end
    end
    params
end

function (g::Graph)(xs::NamedTuple)
    temps = Array{Any}(undef, length(g.nodes))
    for (name,id) in g.inputids
        temps[id] = xs[name]
    end
    for i = 1:length(g.nodes)
        isassigned(temps,i) && continue
        node = g.nodes[i]
        args = map(node.args) do arg
            isa(arg,NodeId) ? temps[arg.id] : arg
        end
        temps[i] = node.f(args...)
    end
    map(id -> temps[id], g.outputids)
end
