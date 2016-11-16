export compile

type GraphNode
    f
    args::Tuple
end

GraphNode(f, args...) = GraphNode(f, args)

type Graph
    nodes::Vector{GraphNode}
    inputs::Vector{Int}
end

function compile(output::GraphNode, inputs::GraphNode...)
    nodes = topsort(output)
    dict = ObjectIdDict()
    nodes = map(nodes) do n
        args = map(n.args) do arg
            typeof(arg) == GraphNode ? dict[arg] : arg
        end
        GraphNode(n.f, args)
    end
    inputs = map(x -> dict[x], inputs)
    Graph(nodes, inputs)
end

function (g::Graph)(x::Var)

end

function topsort(top::GraphNode)
    sorted = GraphNode[]
    dict = ObjectIdDict()
    function visit(node::GraphNode)
        haskey(dict, node) && return
        dict[node] = node
        for arg in node.args
            typeof(arg) == GraphNode && visit(arg)
        end
        push!(sorted, node)
    end
    visit(top)
    sorted
end
