export Graph, Node

mutable struct Node
    f
    args::Tuple
    name::String
    id::Int
end

Node(f=nothing; name="") = Node(f, (), name)
Node(f, args, name) = Node(f, args, name, 0)

struct Graph
    nodes::Vector{Node} # topological order
    dict::Dict{String,Node}
    inputs::Tuple
    outputs::Tuple
    backend
end

function Graph(inputs, outputs; backend="CPU")
    isa(inputs,Tuple) || (inputs = (inputs,))
    isa(outputs,Tuple) || (outputs = (outputs,))
    nodes = topsort(outputs...)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))
    dict = Dict{String,Node}()
    for i = 1:length(nodes)
        node = nodes[i]
        @assert node.id == 0
        node.id = i
        isempty(node.name) || (dict[node.name] = node)
    end
    inputs = map(x -> node2id[x], inputs)
    outputs = map(x -> node2id[x], outputs)

    if backend != "CPU"
        for node in nodes
            isa(node.f,Var) && isparam(node.f) && setbackend!(node.f,backend)
        end
    end
    Graph(nodes, dict, inputs, outputs, backend)
end

Base.getindex(g::Graph, i::Int) = g.nodes[i]
Base.getindex(g::Graph, s::String) = g.dict[s]

function (g::Graph)(xs...)
    @assert length(xs) == length(g.inputs)
    temps = Array{Any}(length(g.nodes))
    for i = 1:length(xs)
        temps[g.inputs[i]] = xs[i]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        if isempty(node.args)
            isassigned(temps,i) || (temps[i] = node.f)
        else
            args = map(node.args) do arg
                isa(arg,Node) ? temps[arg.id] : arg
            end
            temps[i] = node.f(args...)
        end
    end
    if length(g.outputs) == 1
        temps[g.outputs[1]]
    else
        map(id -> temps[id], g.outputs)
    end
end

#=
macro graph(input, output)
    if isa(input, Symbol)
        input = Expr(:tuple, input)
    end
    input.head == :tuple || throw("not tuple")
    exp = Expr(:block)
    for arg in input.args
        e = Expr(:(=), arg, Node()) # x = Node(), etc.
        push!(exp.args, e)
    end
    quote
        $(esc(exp))
        x = $(esc(input))
        y = $(esc(output))
        isa(y,Node) && (y = (y,))
        Graph(x, y)
    end
end
=#
