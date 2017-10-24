export Graph, Node
export getparams

mutable struct Node
    f
    args::Tuple
    name::String
end

Node(; name="") = Node(nothing, (), name)

struct NodeId
    id::Int
end

struct Graph
    nodes::Vector{Node} # topological order
    dict::Dict{String,Node}
    input::Tuple
    output::Tuple
end

function Graph(; input=(), output=())
    isa(input,Tuple) || (input = (input,))
    isa(output,Tuple) || (output = (output,))
    nodes = topsort(output...)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))
    dict = Dict{String,Node}()
    for i = 1:length(nodes)
        node = nodes[i]
        isempty(node.name) || (dict[node.name] = node)
        args = map(node.args) do a
            isa(a,Node) ? NodeId(node2id[a]) : a
        end
        nodes[i] = Node(node.f, args..., name=node.name)
    end
    input = map(x -> node2id[x], input)
    output = map(x -> node2id[x], output)
    Graph(nodes, dict, input, output)
end

Base.getindex(g::Graph, i::Int) = g.nodes[i]
Base.getindex(g::Graph, s::String) = g.dict[s]

"""
    getparams(g::Graph)

Get parameters from Graph `g`.
"""
function getparams(g::Graph)
    dict = Dict{Var,Var}()
    for node in g.nodes
        for arg in node.args
            isa(arg,Var) && isparam(arg) && (dict[arg]=arg)
        end
    end
    collect(keys(dict))
end

function (g::Graph)(xs...)
    @assert length(xs) == length(g.input)
    temps = Array{Any}(length(g.nodes))
    for i = 1:length(xs)
        temps[g.input[i]] = xs[i]
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
    o = map(i -> temps[i], g.output)
    length(o) == 1 ? o[1] : o
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
