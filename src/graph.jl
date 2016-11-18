export compile

type Graph
    nodes::Vector{Var}
    args::Vector{Int}
    f
end

Base.getindex(g::Graph, key::Int) = g.nodes[key]
Base.setindex!(g::Graph, value::Var, key::Int) = g.nodes[key] = value

function compile(output::Var, inputs::Var...)
    @assert output.data == nothing
    all(v -> isempty(v.args) && v.data == nothing, inputs) || throw("Invalid inputs.")
    nodes = topsort(output)
    count(n -> isempty(n.args), nodes) == length(inputs) || throw("Wrong number of inputs.")
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))

    calls = Array{Any}(length(nodes))
    for i = 1:length(nodes)
        node = nodes[i]
        if isempty(node.args)
            calls[i] = node.data == nothing ? gensym() : node
        else
            args = map(node.args) do arg
                typeof(arg) == Var ? Var(node2id[arg]) : arg
            end
            nodes[i] = Var(nothing, args)

            args = map(node.args) do arg
                typeof(arg) == Var ? calls[node2id[arg]] : arg
            end
            calls[i] = Expr(:call, args...)
        end
    end

    args = map(v -> node2id[v], inputs)
    syms = map(a -> calls[a], args)
    expr = Expr(:->, Expr(:tuple, syms...), calls[end])
    f = eval(expr)
    Graph(nodes, [args...], f)
end

(g::Graph)(xs::Var...) = g.f(xs...)

h5convert(g::Graph) = Dict("nodes"=>g.nodes, "inputs"=>g.inputs)
h5convert(::Type{Graph}, x) = Graph(x["nodes"], x["inputs"])

#=
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
=#
