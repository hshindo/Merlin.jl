export Graph

type NodeId
    value::Int
end

h5convert(x::NodeId) = x.value
h5convert(::Type{NodeId}, x) = NodeId(x)

type Graph
    nodes::Vector{Var}
    inputs::Vector{Int}
    outputs::Vector{Int}
    f
end
Graph(nodes, inputs, outputs) = Graph(nodes, inputs, outputs, compile(nodes,inputs,outputs))

function Graph(inputs, outputs)
    all(v -> isvoid(v.data), inputs) || throw("all input data must be Var().")
    nodes = topsort(outputs...)
    node2id = ObjectIdDict(nodes[i]=>i for i=1:length(nodes))
    nodes = map(nodes) do node
        isempty(node.args) && return node
        args = map(node.args) do arg
            isa(arg,Var) ? NodeId(node2id[arg]) : arg
        end
        Var(nothing, node.f, args)
    end
    inputs = Int[node2id[x] for x in inputs]
    outputs = Int[node2id[x] for x in outputs]
    f = compile(nodes, inputs, outputs)
    Graph(nodes, inputs, outputs)
end

function compile(nodes, inputs, outputs)
    calls = []
    for node in nodes
        if isempty(node.args)
            push!(calls, isvoid(node.data) ? gensym() : node)
        else
            nargs = map(node.args) do arg
                isa(arg, NodeId) ? calls[arg.value] : arg
            end
            push!(calls, Expr(:call, node.f, nargs...))
        end
    end
    inputs = map(x -> calls[x], inputs)
    outputs = map(x -> calls[x], outputs)
    e = length(outputs) == 1 ? outputs[1] : Expr(:tuple,outputs...)
    expr = Expr(:->, Expr(:tuple, inputs...), e)
    eval(expr)
end

function (g::Graph)(args::Var...)
    length(args) == length(g.inputs) || throw("input length error.")
    g.f(args...)
end

h5convert(g::Graph) = Dict("nodes"=>g.nodes, "inputs"=>g.inputs, "outputs"=>g.outputs)
h5convert(::Type{Graph}, x) = Graph(x["nodes"], x["inputs"], x["outputs"])

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
