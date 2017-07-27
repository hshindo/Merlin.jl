export Graph, Node, @graph

struct Node
    f
    args

    Node(f, args...) = new(f, args)
end
Node() = Node(nothing)

type NodeId
    id::Int
end

struct Graph
    nodes::Vector{Node} # topological order
    inputs::Vector{Int}
    outputs::Vector{Int}
end

function Graph(inputs::Tuple{Vararg{Node}}, outputs::Tuple{Vararg{Node}})
    nodes = topsort(outputs...)
    node2id = ObjectIdDict(nodes[i]=>i for i=1:length(nodes))
    nodes = map(nodes) do node
        args = map(node.args) do arg
            isa(arg,Node) ? NodeId(node2id[arg]) : arg
        end
        Node(node.f, args...)
    end
    inputs = [map(x -> node2id[x], inputs)...]
    outputs = [map(x -> node2id[x], outputs)...]
    Graph(nodes, inputs, outputs)
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

function (g::Graph)(inputs::Var...)
    vars = Array{Var}(length(g.nodes))
    for i = 1:length(inputs)
        vars[g.inputs[i]] = inputs[i]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        if isempty(node.args)
            isassigned(vars,i) || (vars[i] = node)
        else
            args = map(node.args) do arg
                isa(arg,NodeId) ? vars[arg.id] : arg
            end
            vars[i] = node.f(args...)
        end
    end
    outputs = map(id -> vars[id], g.outputs)
    length(outputs) == 1 ? outputs[1] : outputs
end
