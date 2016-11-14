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
    inputs::Vector{Int}
    f
end

Graph(nodes, inputs) = Graph(nodes, inputs, compile(nodes))

Base.getindex(g::Graph, key::Int) = g.nodes[key]
Base.setindex!(g::Graph, value::Var, key::Int) = g.nodes[key] = value

function (g::Graph)(xs::Var...)
    for x in xs
        x.data == nothing && return Var(nothing, [g, xs...], nothing)
    end
    g.f(xs...)
end
(g::Graph)(xs...) = g(map(constant,xs)...)

function Graph(top::Var, inputs::Var...)
    @assert top.data == nothing
    nodes = topsort(top)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))
    for node in nodes
        # convert var arg to Var(id)
        node.args = map(node.args) do arg
            typeof(arg) == Var ? Var(node2id[arg],nothing) : arg
        end
    end
    inputs = Int[node2id[inputs[i]] for i=1:length(inputs)]
    Graph(nodes, inputs)
end

function compile(nodes::Vector{Var})
    calls = []
    for node in nodes
        if isempty(node.args)
            x = node.data == nothing ? gensym() : node
            push!(calls, x)
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

h5convert(g::Graph) = Dict("nodes"=>g.nodes, "inputs"=>g.inputs)
h5convert(::Type{Graph}, x) = Graph(x["nodes"], x["inputs"])
