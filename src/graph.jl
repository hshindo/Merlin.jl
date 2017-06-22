export Graph, compile

type Graph
    nodes::Vector{Var} # topological order
    inputs::Tuple{Vararg{Int}}
    outputs::Tuple{Vararg{Int}}
end

type VarId
    id::Int
end

function compile(inputs::Tuple, outputs::Tuple)
    nodes = topsort(outputs...)
    node2id = ObjectIdDict(nodes[i]=>i for i=1:length(nodes))
    nodes = map(nodes) do node
        args = map(node.args) do arg
            isa(arg,Var) ? VarId(node2id[arg]) : arg
        end
        Var(node.data, node.batchdims, args)
    end
    inputs = map(x -> node2id[x], inputs)
    outputs = map(x -> node2id[x], outputs)
    Graph(nodes, inputs, outputs)
end
compile(input::Var, outputs) = compile((input,), outputs)
compile(inputs, output::Var) = compile(inputs, (output,))

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
            f = node[1]
            args = ntuple(length(node.args)-1) do i
                arg = node[i+1]
                isa(arg,VarId) ? vars[arg.id] : arg
            end
            vars[i] = f(args...)
        end
    end
    outputs = map(id -> vars[id], g.outputs)
    length(outputs) == 1 ? outputs[1] : outputs
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
