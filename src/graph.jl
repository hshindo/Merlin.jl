export Graph, compile

type Graph
    nodes::Vector{Var} # topological order
    inputs::Vector{Int}
    output::Int
end

type VarId
    id::Int
end

function compile(output::Var, inputs::Var...)
    nodes = topsort(output)
    node2id = ObjectIdDict(nodes[i]=>i for i=1:length(nodes))
    nodes = map(nodes) do node
        args = map(node.args) do arg
            isa(arg,Var) && return VarId(node2id[arg])
            isa(arg,Vector{Var}) && return map(x -> VarId(node2id[x]),arg)
            arg
        end
        Var(node.data, node.f, args)
    end
    inputs = map(x -> node2id[x], inputs)
    output = node2id[output]
    Graph(nodes, inputs, output)
end

function (g::Graph)(inputs::Union{Var,Vector{Var}}...)
    vars = Array{Var}(length(g.nodes))
    for i = 1:length(inputs)
        vars[g.inputs[i]] = inputs[i]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        if isempty(node.args)
            isdefined(vars,i) || (vars[i] = node)
        else
            xs = map(node.args) do arg
                isa(arg,VarId) && return vars[arg.id]
                isa(arg,Vector{VarId}) && return map(x -> vars[x.id], arg)
                arg
            end
            vars[i] = node.f(xs...)
        end
    end
    vars[end]
end

#=
function (g::Graph)(xs...)
    ys = Array{Any}(length(g.nodes))
    for i = 1:length(xs)
        ys[g.input[i]] = xs[i]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        if isempty(node.args)
            isdefined(ys,i) || (ys[i] = node)
            continue
        end
        xs = map(x -> ys[x], node.args)
        ys[i] = node.f(xs...)
    end

    gg = Graph(ys, g.input, g.output)
    y = Var(ys[end].data, gg, xs)
    y.df! = function df!()
        for i = 1:length(y.f.nodes)
            v = y.f.nodes[i]
            isempty(v.args) && continue
            #all(a -> isvoid(a.grad), v.args) && continue
            isvoid(v.grad) && zerograd!(v)
        end
        for i = length(y.f.nodes):-1:1
            v = y.f.nodes[i]
            isvoid(v.df!) || v.df!()
        end
    end
    y
end
=#

function compile!(top::Var)
    nodes = topsort(top)


    dict1 = ObjectIdDict()
    dict2 = ObjectIdDict()
    for node in nodes
        isa(node.f,typeof(*)) || continue
        vars = get!(dict1, node[1], Var[])
        push!(vars, node[2])
        vars = get!(dict2, node[2], Var[])
        push!(vars, node[1])
    end

end
