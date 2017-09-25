export Graph, Node, @graph, getparams

mutable struct Node
    f
    args::Tuple
    name::String

    Node(f, args...; name="") = new(f, args, name)
end
Node() = Node(nothing)

struct NodeId
    id::Int
end

struct Graph
    nodes::Vector{Node} # topological order
    inputs::Vector{Int}
    outputs::Vector{Int}
end

function Graph(inputs::Tuple{Vararg{Node}}, outputs::Tuple{Vararg{Node}})
    length(outputs) == 1 || throw("Not implemented yet.")
    nodes = topsort(outputs[1])
    node2id = ObjectIdDict(nodes[i]=>i for i=1:length(nodes))

    nodes = map(nodes) do node
        args = map(node.args) do arg
            isa(arg,Node) ? NodeId(node2id[arg]) : arg
        end
        Node(node.f, args..., name=node.name)
    end
    inputs = [map(x -> node2id[x], inputs)...]
    outputs = [map(x -> node2id[x], outputs)...]
    Graph(nodes, inputs, outputs)
end

function getparams(g::Graph)
    params = Var[]
    for n in g.nodes
        for arg in n.args
            isa(arg,Var) && isparam(arg) && push!(params,arg)
        end
    end
    params
end

"""
```julia
f = @graph n begin
    Node(relu, n)
end
```
"""
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

function (g::Graph)(xs...)
    temps = Array{Any}(length(g.nodes))
    for i = 1:length(xs)
        temps[g.inputs[i]] = xs[i]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        if isempty(node.args)
            isassigned(temps,i) || (temps[i] = node)
        else
            args = map(node.args) do arg
                isa(arg,NodeId) ? temps[arg.id] : arg
            end
            temps[i] = node.f(args...)
        end
    end
    outputs = map(id -> temps[id], g.outputs)
    length(outputs) > 1 && throw("Not implemented yet.")
    v = outputs[1]
    v
end

Base.size(x::Node) = Node(size, x)
Base.size(x::Node, i::Int) = Node(size, x, i)
Base.length(x::Node) = Node(length, x)
Base.ndims(x::Node) = Node(ndims, x)
