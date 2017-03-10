export Graph

type VarId
    value::Int
end

type Graph
    nodes::Vector{Var}
    inputs::Vector{Int}
    f
end

function Graph(output::Var, inputs::Tuple{Vararg{Var}})
    nodes = topsort(output)
    node2id = ObjectIdDict(nodes[i]=>i for i=1:length(nodes))
    nodes = map(nodes) do node
        isempty(node.args) && return node
        args = map(node.args) do arg
            isa(arg, Var) ? VarId(node2id[arg]) : arg
        end
        Var(nothing, node.f, args)
    end
    inputs = Int[node2id[x] for x in inputs]
    Graph(nodes, inputs, nothing)
end

function (g::Graph)(args::Var...)
    g.f == nothing && (g.f = compile(g))
    g.f(args...)
end

function compile(g::Graph)
    calls = []
    for node in g.nodes
        if isempty(node.args)
            push!(calls, isvoid(node.data) ? gensym() : node)
        else
            args = map(node.args) do arg
                isa(arg,VarId) ? calls[arg.value] : arg
            end
            push!(calls, Expr(:call, node.f, args...))
        end
    end
    inputs = map(x -> calls[x], g.inputs)
    expr = Expr(:->, Expr(:tuple, inputs...), calls[end])
    eval(expr)
end

h5convert(g::Graph) = Dict("nodes"=>g.nodes, "args"=>g.args)
h5convert(::Type{Graph}, x) = Graph(x["nodes"], x["args"])

#=
macro graph(expr)
    (expr.head == :function || expr.head == :(=)) || throw("Invalid @graph.")
    args, vars = [], []
    for arg in expr.args[1].args
        if isa(arg, Expr)
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
=#
