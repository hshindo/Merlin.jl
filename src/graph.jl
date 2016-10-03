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

Graph(nodes::Vector{Var}) = Graph(nodes, compile(nodes))

(g::Graph)(x...) = g.f(x...)

function Graph(top::Var)
    @assert top.data == nothing
    nodes = topsort(top)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))
    for node in nodes
        node.args = map(node.args) do arg
            typeof(arg) == Var ? Var(node2id[arg],nothing) : arg
        end
    end
    Graph(nodes)
end

function compile(nodes::Vector{Var})
    calls = []
    for node in nodes
        if isempty(node.args)
            push!(calls, gensym())
        else
            args = map(node.args) do arg
                typeof(arg) == Var ? calls[arg.data] : arg
            end
            push!(calls, Expr(:call, args...))
        end
    end
    syms = filter(x -> typeof(x) == Symbol, calls)
    expr = Expr(:->, Expr(:tuple, syms...), calls[end])
    eval(expr)
end

h5object(g::Graph) = h5object(g.nodes)
h5load(::Type{Graph}, x) = Graph(h5load(Vector{Var},x))
