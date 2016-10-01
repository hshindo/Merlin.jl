export @graph, Graph

"""
    @graph
"""
macro graph(expr)
    (expr.head == :function || expr.head == :(=)) || throw("Invalid @graph.")

    args = Expr(:vect)
    vars = Expr(:vect) # function arguments typed as `::Var`
    for arg in expr.args[1].args
        if typeof(arg) == Expr
            argname, argtype = arg.args[1], arg.args[2]
            argtype == :Var && push!(vars.args, argname)
            push!(args.args, argname)
        else
            push!(args.args, arg)
        end
    end
    length(vars.args) == 0 && throw("The @graph function must contain at least one argument typed as `::Var`.")

    # Add function body to
    body = expr.args[2].args # function body
    for v in vars.args
        unshift!(body, :($v.data == nothing && return Var(nothing,$vars,$args,nothing)))
    end
    :($expr)
end

type GraphNodeId
    id::Int
end

to_hdf5(x::GraphNodeId) = x.id
from_hdf5(::Type{GraphNodeId}, x) = GraphNodeId(x)

"""
    Graph
"""
type Graph
    nodes::Vector{Var}
    f
end

(g::Graph)(x...) = g.f(x...)

function Graph(top::Var)
    nodes = topsort(top)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))
    calls = []

    for node in nodes
        if isempty(node.args)
            push!(calls, gensym())
        else
            for i = 1:length(node.f)
                x = node.f[i]
                typeof(x) == Var && (node.f[i] = GraphNodeId(node2id[x]))
            end
            args = map(node.f) do arg
                typeof(arg) == GraphNodeId ? calls[arg.id] : arg
            end
            push!(calls, Expr(:call, args...))
        end
    end
    syms = filter(x -> typeof(x) == Symbol, calls)
    expr = Expr(:->, Expr(:tuple, syms...), calls[end]) # create anonymous function
    f = eval(expr)
    Graph(nodes, f)
end

function to_hdf5(g::Graph)
    dict = Dict()
    for i = 1:length(g.nodes)
        v = g.nodes[i]
        dict[i] = v.f
    end
    dict
end

function from_HDF5(::Type{Graph}, x::Dict)
    nodes = Array(Var, length(x))
    for (k,v) in x
        var = Var()
        var.f = v
        nodes[parse(Int,k)] = var
    end
    Graph(nodes, nothing)
end
