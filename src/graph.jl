export compile, @graph

type GraphNode <: AbstractNode
    args::Vector

    GraphNode(args...) = new(Any[args...])
end

function compile(top::GraphNode, syms::Symbol...)
    nodes = topsort(top)
    dict = ObjectIdDict()
    for node in nodes
        args = map(node.args) do n
            #typeof(n) == Symbol && (syms[n] = n)
            typeof(n) == GraphNode ? dict[n] : n
        end
        dict[node] = Expr(:call, args...)
    end
    expr = Expr(:->, Expr(:tuple, syms...), dict[top]) # create anonymous function
    eval(expr)
end

macro graph(expr)
    bottomup(expr) do ex
        if ex.head == :call
            unshift!(ex.args, :(Merlin.GraphNode))
        end
    end
    esc(expr)
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

macro graph2(expr)
    isnode(ex) = typeof(ex) == Expr && ex.head == :call && ex.args[1] == :GraphNode
    function conv(ex::Expr)
        for a in ex.args
            typeof(a) == Expr && conv(a)
        end
        ex.head == :call || return
        any(isnode, ex.args) && unshift!(ex.args, :GraphNode)
    end
    conv(expr)
    expr
    #quote
    #    Graph($(esc(expr)))
    #end
end
=#
