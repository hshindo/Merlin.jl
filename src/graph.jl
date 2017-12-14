export Graph

#=
mutable struct Node
    f
    args::Tuple
    name::String
    id::Int
end

Node(name::String) = Node(nothing, (), name)
Node(f, args, name) = Node(f, args, name, 0)
=#

struct Graph
    nodes::Vector{Var} # topological order
    node2id::Dict{Var,Int}
    name2id::Dict{String,Int}
    output::Tuple
end

function Graph(output::Var...)
    nodes = topsort(output...)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))
    name2id = Dict{String,Int}()
    for i = 1:length(nodes)
        node = nodes[i]
        isempty(node.name) || (name2id[node.name] = i)
    end
    output = map(x -> node2id[x], output)
    Graph(nodes, node2id, name2id, output)
end

Base.getindex(g::Graph, i::Int) = g.nodes[i]
Base.getindex(g::Graph, s::String) = g.node2id[s]

function (g::Graph)(xs::Pair...)
    temps = Array{Any}(length(g.nodes))
    for x in xs
        id = g.name2id[x[1]]
        temps[id] = x[2]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        if isempty(node.args)
            isassigned(temps,i) || (temps[i] = node)
        else
            args = ntuple(length(node.args)-1) do i
                arg = node.args[i+1]
                if isa(arg, Var)
                    temps[g.node2id[arg]]
                elseif isa(arg, Vector{Var})
                    map(x -> temps[g.node2id[x]], arg)
                else
                    arg
                end
            end
            temps[i] = node.args[1](args...)
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
