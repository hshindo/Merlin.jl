export @graph, compile

"""
    @graph
"""
macro graph(expr)
    (expr.head == :function || expr.head == :(=)) || throw("Invalid @graph.")

    args = Expr(:vect)
    vars = Expr(:vect) # function arguments typed as `::Var`
    for arg in expr.args[1].args
        if typeof(arg) == Expr
            argname, argtype = arg.args[1], arg.args[2]
            argtype == :Var && push!(vars.args, argname)
            push!(args.args, argname)
        else
            push!(args.args, arg)
        end
    end
    length(vars.args) == 0 && throw("The @graph function must contain at least one argument typed as `::Var`.")

    # Add function body to
    body = expr.args[2].args # function body
    for v in vars.args
        unshift!(body, :($v.data == nothing && return Var(nothing,$vars,$args,nothing)))
    end
    :($expr)
end

"""
    Graph
"""
type Graph
    nodes::Vector{Var}
    f
end

(g::Graph)(x...) = g.f(x...)

function compile(top::Var)
    nodes = topsort(top)
    srcs = filter(n -> isempty(n.args), nodes)
    dict = ObjectIdDict()
    syms = map(srcs) do x
        s = gensym()
        dict[x] = s
        s
    end
    for node in nodes
        isempty(node.args) && continue
        args = map(node.f) do arg
            typeof(arg) == Var ? dict[arg] : arg
        end
        dict[node] = Expr(:call, args...)
    end
    expr = Expr(:->, Expr(:tuple, syms...), dict[nodes[end]]) # create anonymous function
    f = eval(expr)
    Graph(nodes, f)
end

function to_hdf5(g::Graph)
    dict = Dict()
    for i = 1:length(g.nodes)
        v = g.nodes[i]
        dict[i] = v.f
    end
    dict
end

function h5convert(g::Graph)
    dict = h5dict(Graph)
    argdict = ObjectIdDict()
    for i = 1:length(x)
        d = Dict{String,Any}()
        dict[string(i)] = d
        for j = 1:length(x[i])
            n = x[i][j]
            if typeof(n) == GraphNode
                d[string(j)] = Dict("#NODE"=>argdict[n])
            else
                d[string(j)] = h5convert(n)
            end
        end
        argdict[x[i]] = i
    end
    dict
end
