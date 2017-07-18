export Graph, @graph

struct Graph
    nodes::Vector{Var} # topological order
    inputs::Vector{Int}
    outputs::Vector{Int}
end

type VarId
    id::Int
end

function Graph(inputs::Union{Var,Vector{Var}}, outputs::Union{Var,Vector{Var}})
    inputs = isa(inputs,Var) ? [inputs] : inputs
    outputs = isa(outputs,Var) ? [outputs] : outputs
    nodes = topsort(outputs...)
    node2id = ObjectIdDict(nodes[i]=>i for i=1:length(nodes))
    nodes = map(nodes) do node
        args = map(node.args) do arg
            isa(arg,Var) ? VarId(node2id[arg]) : arg
        end
        Var(node.data, node.f, args, node.df!, node.grad)
    end
    inputs = map(x -> node2id[x], inputs)
    outputs = map(x -> node2id[x], outputs)
    Graph(nodes, inputs, outputs)
end

"""
```julia
f = @graph x begin
    relu(x)
end
```
"""
macro graph(input, output)
    if isa(input, Expr)
        input.head == :tuple || throw("not tuple")
        exp = Expr(:block)
        for arg in input.args
            e = Expr(:(=), arg, Var()) # x = Var(), etc.
            push!(exp.args, e)
        end
    else
        exp = Expr(:(=), input, Var())
    end
    quote
        $(esc(exp))
        Graph($(esc(input)), $(esc(output)))
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
                isa(arg,VarId) ? vars[arg.id] : arg
            end
            vars[i] = node.f(args...)
        end
    end
    outputs = map(id -> vars[id], g.outputs)
    length(outputs) == 1 ? outputs[1] : outputs
end

readas(::Type{Graph}, d::Dict) = Graph(d["nodes"], d["inputs"], d["outputs"])
writeas(g::Graph) = Dict("nodes"=>g.nodes, "inputs"=>g.inputs, "outputs"=>g.outputs)

#=
function (g::Graph)(inputs::Vector{Var})
    ys = Var[]
    for x in inputs
        y = g(x)
        push!(ys, y)
    end
    cat(ndims(ys[1].data), ys...)
end
=#
