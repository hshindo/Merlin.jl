export Graph, Node
export compile, getparams

mutable struct Node
    f
    args::Tuple

    Node(f, args...) = new(f, args)
end

Node() = Node(nothing)

Base.getindex(n::Node, i::Int) = n.args[i]

function compile(n::Node, backend::Backend)
    f = compile(n.f, backend)
    args = map(n.args) do a
        compile(a, backend)
    end
    Node(f, args...)
end

struct NodeId
    id::Int
end

struct Graph
    nodes::Vector{Node} # topological order
    inputs::Tuple
    outputs::Tuple
end

function Graph(inputs, outputs)
    isa(inputs,Tuple) || (inputs = (inputs,))
    isa(outputs,Tuple) || (outputs = (outputs,))
    nodes = topsort(outputs...)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))

    nodes = map(nodes) do node
        args = map(node.args) do arg
            isa(arg,Node) ? NodeId(node2id[arg]) : arg
        end
        Node(node.f, args...)
    end
    inputs = map(x -> node2id[x], inputs)
    outputs = map(x -> node2id[x], outputs)
    Graph(nodes, inputs, outputs)
end

Base.getindex(g::Graph, i::Int) = g.nodes[i]
Base.getindex(g::Graph, s::String) = g.dict[s]

function getparams(g::Graph)
    params = Var[]
    for node in g.nodes
        for arg in node.args
            isa(arg,Var) && isparam(arg) && push!(params,arg)
        end
    end
    params
end

function compile(g::Graph, backend::Backend)
    nodes = map(n -> compile(n,backend), g.nodes)
    Graph(nodes, g.inputs, g.outputs)
end

function (g::Graph)(xs...)
    @assert length(xs) == length(g.inputs)
    temps = Array{Any}(length(g.nodes))
    for i = 1:length(xs)
        temps[g.inputs[i]] = xs[i]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        isvoid(node.f) && continue
        args = map(node.args) do arg
            isa(arg,NodeId) ? temps[arg.id] : arg
        end
        temps[i] = node.f(args...)
    end
    if length(g.outputs) == 1
        temps[g.outputs[1]]
    else
        map(id -> temps[id], g.outputs)
    end
end
