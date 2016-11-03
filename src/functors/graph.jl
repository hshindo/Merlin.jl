type Graph <: Functor
    nodes::Vector{Var}
end

function Graph(top::Var)
    nodes = topsort(top)
    Graph(nodes)
end

Base.getindex(g::Graph, key::Int) = g.nodes[key]
Base.setindex!(g::Graph, value, key::Int) = g.nodes[key] = value

(g::Graph)(x::Var) = forward(g, x)

function forward!(g::Graph, v::Var)::Void
    g[1].args = v.args
    for n in g.nodes
        forward!(n.f, n)
    end
    v.data = g[end].data
end

function backward!(g::Graph, v::Var)
    g[end].grad = v.grad
    g[1].grad = v[1].grad
    for i = length(g.nodes):-1:1
        backward!(g[i].f, g[i])
    end
end
