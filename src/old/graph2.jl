export Graph, Node
export compile, getparams

mutable struct Node
    args::Tuple

    Node(args...) = new(args)
end

Base.getindex(n::Node, i::Int) = n.args[i]

call(f, args...) = f(args...)

struct NodeId
    id::Int
end

struct Graph
    nodes::Vector{Node} # topological order
    dict::Dict{String,Node}
    inputs::Tuple
    outputs::Tuple
end

function Graph(inputs, outputs)
    isa(inputs,Tuple) || (inputs = (inputs,))
    isa(outputs,Tuple) || (outputs = (outputs,))
    nodes = topsort(outputs...)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))

    dict = Dict{String,Node}()
    nodes = map(nodes) do node
        args = map(node.args) do arg
            isa(arg,Node) ? NodeId(node2id[arg]) : arg
        end
        newnode = Node(args...)
        # isempty(node.name) || (dict[node.name] = newnode)
        newnode
    end
    inputs = map(x -> node2id[x], inputs)
    outputs = map(x -> node2id[x], outputs)
    Graph(nodes, dict, inputs, outputs)
end

Base.getindex(g::Graph, i::Int) = g.nodes[i]
Base.getindex(g::Graph, s::String) = g.dict[s]

function getparams(g::Graph)
    params = Var[]
    for node in g.nodes
        if length(node.args) == 1 && isa(node[1],Var)
            push!(params, node[1])
        end
    end
    params
end

function compile(g::Graph, backend)
    @assert backend == "CUDA:0"
    for node in g.nodes
        isempty(node.args) && continue
        if length(node.args) == 1
            setbackend!(node[1],backend)
        else
            node.args = cucompile(node.args...)
        end
    end
end

function (g::Graph)(xs...)
    @assert length(xs) == length(g.inputs)
    temps = Array{Any}(length(g.nodes))
    for i = 1:length(xs)
        temps[g.inputs[i]] = xs[i]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        if isempty(node.args)
            continue
        elseif length(node.args) == 1
            # isassigned(temps,i) || (
            temps[i] = node[1]
        else
            args = map(node.args) do arg
                isa(arg,NodeId) ? temps[arg.id] : arg
            end
            temps[i] = call(args...)
        end
    end
    if length(g.outputs) == 1
        temps[g.outputs[1]]
    else
        map(id -> temps[id], g.outputs)
    end
end
