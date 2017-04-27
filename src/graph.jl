export Graph

type Graph
    nodes::Vector{Var} # bottomup order
    inids::Vector{Int}
    outids::Vector{Int}
    f
end
Graph(nodes, inputs, outputs) = Graph(nodes, inputs, outputs, nothing)

function Graph(inputs::Vector{Var}, outputs::Vector{Var})
    nodes = topsort(outputs...)
    node2id = ObjectIdDict(nodes[i]=>i for i=1:length(nodes))
    nodes = map(nodes) do node
        isempty(node.args) && return node
        args = map(node.args) do arg
            isa(arg, Var) && return NodeId(node2id[arg])
            arg
        end
        Var(nothing, node.f, args)
    end
    inids = Int[node2id[x] for x in inputs]
    outids = Int[node2id[x] for x in outputs]
    Graph(nodes, inids, outids)
end

function update!(g::Graph, opt)
    for v in g.nodes
        if isparam(v)
            #println(v.grad)
            #throw("")
            opt(v.data, v.grad)
        end
    end
end

function (g::Graph)(args::Var...)
    length(args) == length(g.inids) || throw("input length error.")
    g.f == nothing && (g.f = compile(g))
    g.f(args...)
end

#=
function (g::Graph)(args::Var...)
    length(args) == length(g.inids) || throw("input length error.")
    g.f == nothing && (g.f = compile(g))
    outputs = g.f(args...)

    function backward!(gy, ppp...)
        outputs[end].grad = gy
        for v in outputs
            isvoid(v.grad) && !isempty(v.args) && zerograd!(v)
        end
        for i = length(outputs):-1:1
            v = outputs[i]
            isvoid(v.df) && continue
            args = Any[v.grad]
            for arg in v.args
                isa(arg, Var) && push!(args, arg.grad)
                isa(arg, Vector{Var}) && push!(args, map(a -> a.grad, arg))
            end
            v.df(args...)
        end
    end
    Var(outputs[end].data, g, args, backward!)
end
=#

readas(::Type{Graph}, x) = Graph(x["nodes"], x["inids"], x["outids"])
writeas(g::Graph) = Dict("nodes"=>g.nodes, "inids"=>g.inids, "outids"=>g.outids)

type NodeId
    value::Int
end

readas(::Type{NodeId}, x) = NodeId(x)
writeas(x::NodeId) = x.value

function compile2(g::Graph)
    syms = [gensym() for i=1:length(g.nodes)]
    indict = Dict(id=>id for id in g.inids)
    calls = []
    for i = 1:length(g.nodes)
        haskey(indict, i) && continue
        node = g.nodes[i]
        if isempty(node.args)
            push!(calls, Expr(:(=),syms[i],node))
        else
            nargs = map(node.args) do arg
                isa(arg, NodeId) && return syms[arg.value]
                isa(arg, Vector{NodeId}) && return Expr(:vect, map(x->syms[x.value],arg)...)
                arg
            end
            expr = Expr(:call, node.f, nargs...)
            push!(calls, Expr(:(=),syms[i],expr))
        end
    end

    inputs = map(id -> syms[id], g.inids)
    push!(calls, Expr(:vect, syms...))
    expr = Expr(:->, Expr(:tuple, inputs...), Expr(:block, calls...))
    eval(expr)
end

function compile(g::Graph)
    calls = Array{Any}(length(g.nodes))
    foreach(id -> calls[id] = gensym(), g.inids)
    for i = 1:length(calls)
        node = g.nodes[i]
        if isempty(node.args)
            isdefined(calls, i) || (calls[i] = node)
        else
            nargs = map(node.args) do arg
                isa(arg, NodeId) && return calls[arg.value]
                isa(arg, Vector{NodeId}) && return Expr(:vect, map(x->calls[x.value],arg)...)
                arg
            end
            calls[i] = Expr(:call, node.f, nargs...)
        end
    end

    inputs = map(x -> calls[x], g.inids)
    outputs = map(x -> calls[x], g.outids)
    e = length(outputs) == 1 ? outputs[1] : Expr(:tuple,outputs...)
    expr = Expr(:->, Expr(:tuple, inputs...), e)
    eval(expr)
end
