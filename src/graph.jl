export compile, @graph

type GraphNode <: AbstractNode
    args::Vector

    GraphNode(args...) = new(Any[args...])
end

type Graph
    top::GraphNode
    f
end

@compat (g::Graph)(xs...) = g.f(xs...)

function compile(top::GraphNode, syms::Tuple{Vararg{Symbol}})
    nodes = topsort(top)
    dict = ObjectIdDict()
    for node in nodes
        args = map(node.args) do n
            typeof(n) == GraphNode ? dict[n] : n
        end
        dict[node] = Expr(:call, args...)
    end
    expr = Expr(:->, Expr(:tuple, syms...), dict[top]) # create anonymous function
    eval(expr)
end

macro graph(args, expr)
    bottomup(expr) do ex
        if ex.head == :call
            unshift!(ex.args, :(Merlin.GraphNode))
        end
    end
    quote
        local top = $(esc(expr))
        local f = compile(top, $args)
        Graph(top, f)
    end
end

#=
function to_hdf5(g::Graph)
  d_nodes = Dict()
  for i = 1:length(g.nodes)
    d_nodes[string(i)] = to_hdf5(g.nodes[i])
  end
  d_sym2id = Dict()
  for (k,v) in g.sym2id
    d_sym2id[string(k)] = v
  end
  Dict("Graph" => Dict("nodes" => d_nodes, "sym2id" => d_sym2id))
end
=#
