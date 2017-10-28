export Graph, Node

mutable struct Node
    f
    args::Tuple
    name::String
    id::Int
end

Node(; name="") = Node(nothing, (), name)
Node(f, args, name) = Node(f, args, name, 0)

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
        if node.id != 0
            throw("Node $(node) is already used.")
        end
        node.id = i
        isempty(node.name) || (dict[node.name] = node)
    end
    input = map(x -> node2id[x], input)
    output = map(x -> node2id[x], output)
    Graph(nodes, dict, input, output)
end

Base.getindex(g::Graph, i::Int) = g.nodes[i]
Base.getindex(g::Graph, s::String) = g.dict[s]

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
                isa(arg,Node) ? temps[arg.id] : arg
            end
            temps[i] = node.f(args...)
        end
    end
    o = map(i -> temps[i], g.output)
    length(o) == 1 ? o[1] : o
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

#function convert(::Type{H5Object}, x::Graph)
#    H5Object(typeof(x), x)
#end
