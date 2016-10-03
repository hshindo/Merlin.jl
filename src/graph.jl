export @graph, Graph

"""
    @graph
"""
macro graph(expr)
    (expr.head == :function || expr.head == :(=)) || throw("Invalid @graph.")
    args, vars = [], []
    for arg in expr.args[1].args
        if typeof(arg) == Expr
            aname, atype = arg.args[1], arg.args[2]
            atype == :Var && push!(vars, aname)
            push!(args, aname)
        else
            push!(args, arg)
        end
    end
    length(vars) == 0 && throw("The @graph function must contain at least one argument typed as `::Var`.")

    args[1] = QuoteNode(args[1])
    args = Expr(:vect, args...)
    body = expr.args[2].args
    for v in vars
        unshift!(body, :($v.data == nothing && return Var(nothing,$args,nothing,nothing)))
    end
    :($expr)
end

"""
    Graph
"""
type Graph
    nodes::Vector{Var}
    f
end

(g::Graph)(x...) = g.f(x...)

function Graph(top::Var)
    @assert top.data == nothing
    nodes = topsort(top)
    foreach(i -> nodes[i].data = i, 1:length(nodes))

    calls = []
    for node in nodes
        if isempty(node.args)
            push!(calls, gensym())
        else
            args = map(node.args) do arg
                typeof(arg) == Var ? calls[arg.data] : arg
            end
            node.args = map(node.args) do arg
                #typeof(arg) == Var ?
            end
            push!(calls, Expr(:call, args...))
        end
    end
    syms = filter(x -> typeof(x) == Symbol, calls)
    expr = Expr(:->, Expr(:tuple, syms...), calls[end]) # create anonymous function
    Graph(nodes, eval(expr))
end

function h5convert(g::Graph)
    dict = Dict()
    for i = 1:length(g.nodes)
        dict[i] = i
    end
    Graph, dict
end

function to_hdf5(g::Graph)
    dict = Dict()
    for i = 1:length(g.nodes)
        v = g.nodes[i]
        args = map(a -> typeof(a) == Var ? constant(a.data) : a, v.args)
        dict[i] = Var(v.data, v.grad, args, nothing, nothing)
    end
    dict
end

function from_hdf5(::Type{Graph}, x::Dict)
    nodes = Array(Var, length(x))
    for (k,v) in x
        var = Var()
        var.f = v
        nodes[parse(Int,k)] = var
    end
    Graph(nodes)
end
