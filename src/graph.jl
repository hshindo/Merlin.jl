export compile

type Graph
    nodes::Vector{Var}
    args::Tuple
    f
end

Graph(nodes, args) = compile!(Graph(nodes,args,nothing))

Base.getindex(g::Graph, key::Int) = g.nodes[key]
Base.setindex!(g::Graph, value::Var, key::Int) = g.nodes[key] = value

function compile(output::Var, inputs::Var...)
    @assert output.data == nothing
    all(v -> isempty(v.args) && v.data == nothing, inputs) || throw("Invalid inputs.")
    nodes = topsort(output)
    #count(n -> isempty(n.args), nodes) == length(inputs) || throw("Wrong number of inputs.")

    # convert Var arg to index (for saving object)
    node2id = Dict(nodes[i]=>i for i=1:length(nodes))
    nodes = map(nodes) do node
        isempty(node.args) && return node
        args = map(node.args) do arg
            typeof(arg) <: Var ? Var(node2id[arg]) : arg
        end
        Var(node.data, node.f, args)
    end
    args = map(x -> node2id[x], inputs)
    Graph(nodes, args)
end

function compile!(g::Graph)
    calls = []
    for node in g.nodes
        if isempty(node.args)
            push!(calls, isvoid(node.data) ? gensym() : node)
        else
            args = map(node.args) do arg
                typeof(arg) <: Var ? calls[arg.data] : arg
            end
            push!(calls, Expr(:call, node.f, args...))
        end
    end
    syms = map(a -> calls[a], g.args)
    expr = Expr(:->, Expr(:tuple, syms...), calls[end])
    g.f = eval(expr)
    g
end

(g::Graph)(xs::Var...) = g.f(xs...)

h5convert(g::Graph) = Dict("nodes"=>g.nodes, "args"=>g.args)
h5convert(::Type{Graph}, x) = Graph(x["nodes"], x["args"])

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
