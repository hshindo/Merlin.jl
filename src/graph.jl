export Graph

type Graph <: Functor
    nodes::Vector{Var} # bottomup order
    input::Tuple
    output::Tuple
end

function Graph(input, output)
    isa(input,Var) && (input = (input,))
    isa(output,Var) && (output = (output,))
    nodes = topsort(output...)
    node2id = ObjectIdDict(nodes[i]=>i for i=1:length(nodes))
    nodes = map(nodes) do node
        args = map(a -> node2id[a], node.args)
        Var(node.data, node.f, args)
    end
    inids = map(x -> node2id[x], input)
    outids = map(x -> node2id[x], output)
    Graph(nodes, inids, outids)
end

function (g::Graph)(xs...)
    ys = Array{Any}(length(g.nodes))
    for i = 1:length(xs)
        ys[g.input[i]] = xs[i]
    end
    for i = 1:length(g.nodes)
        node = g.nodes[i]
        if isempty(node.args)
            isdefined(ys, i) || (ys[i] = node)
            continue
        end
        #println(node.args)
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

function update!(g::Graph, opt)
    for v in g.nodes
        if isa(v.f,Functor)
            update!(v.f,opt)
        elseif isparam(v)
            println("update")
            opt(v.data, v.grad)
        end
    end
end
