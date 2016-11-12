export compile

type GraphNode
    f
    args::Vector
end

type Graph
    nodes::Vector{GraphNode}
end

function (g::Graph)(x::Var)
    

    dfs = Function[]
    g.nodes[1].data = x.data
    for v in g.nodes
        v.f(v)
        push!(dfs, v.df)
    end
    df(gy) = foreach(df -> df(), dfs)
    Var(nodes[end].data, nothing, df, [x])
end

function forward(f::Graph, l::Layer)
    layers = f.layers
    layers[1].args = l.args
    for x in layers
        forward!(x)
    end
    l.data = layers[end].data
end
