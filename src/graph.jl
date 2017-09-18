export Graph, Node, @graph

mutable struct Node
    f
    args::Tuple
    name::String

    Node(f, args...; name="") = new(f, args, name)
end
Node() = Node(nothing)

mutable struct NodeId
    id::Int
end

mutable struct Graph <: Functor
    nodes::Vector{Node} # topological order
    inputs::Vector{Int}
    outputs::Vector{Int}
    params::Vector
end

function Graph(inputs::Tuple{Vararg{Node}}, outputs::Tuple{Vararg{Node}})
    length(outputs) == 1 || throw("Not implemented yet.")
    nodes = topsort(outputs[1])
    node2id = ObjectIdDict(nodes[i]=>i for i=1:length(nodes))
    params = []
    nodes = map(nodes) do node
        isa(node.f,Functor) && push!(params,node.f)
        args = map(node.args) do arg
            isa(arg,Node) ? NodeId(node2id[arg]) : arg
        end
        Node(node.f, args...)
    end
    inputs = [map(x -> node2id[x], inputs)...]
    outputs = [map(x -> node2id[x], outputs)...]
    g = Graph(nodes, inputs, outputs, params)
    compile!(g)
    g
end

function compile!(g::Graph)
    params = Var[]
    for n in g.nodes
        for arg in n.args
            isa(arg,Var) && !isvoid(arg.grad) && push!(params,arg)
        end
    end
    isempty(params) && return

    len = sum(p -> length(p.data), params)
    T = eltype(params[1].data)
    var = zerograd(Array{T}(len))
    i = 1
    for p in params
        subdata = unsafe_wrap(Array, pointer(var.data,i), size(p.data))
        p.data = copy!(subdata, p.data)
        subgrad = unsafe_wrap(Array, pointer(var.grad,i), size(p.data))
        p.grad = copy!(subgrad, p.grad)
        i += length(p.data)
    end
    push!(g.params, var)
end

"""
```julia
f = @graph n begin
    Node(relu, n)
end
```
"""
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

function (g::Graph)(inputs::Var...)
    vars = Array{Var}(length(g.nodes))
    for i = 1:length(inputs)
        vars[g.inputs[i]] = inputs[i]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        if isempty(node.args)
            isassigned(vars,i) || (vars[i] = node)
        else
            args = map(node.args) do arg
                isa(arg,NodeId) ? vars[arg.id] : arg
            end
            vars[i] = node.f(args...)
        end
    end
    outputs = map(id -> vars[id], g.outputs)
    length(outputs) > 1 && throw("Not implemented yet.")
    v = outputs[1]
    Var(v.data, v.batchdims, g, inputs, work=vars)
end

function addgrad!(y::Var, g::Graph, xs::Var...)
    vars = y.work
    vars[end].grad = y.grad
    for v in vars
        !isempty(v.args) && isvoid(v.grad) && zerograd!(v)
    end
    for i = length(vars):-1:1
        v = vars[i]
        addgrad!(v)
        isvoid(v.grad) && continue
    end
end

function update!(g::Graph, opt)
    for p in g.params
        if isa(p, Functor)
            update!(p, opt)
        elseif isa(p, Var)
            opt(p.data, p.grad)
        end
    end
end

Base.size(x::Node) = Node(size, x)
Base.size(x::Node, i::Int) = Node(size, x, i)
Base.length(x::Node) = Node(length, x)
Base.ndims(x::Node) = Node(ndims, x)
