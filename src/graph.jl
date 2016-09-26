export @graph, compile

macro graph(expr)
    (expr.head == :function || expr.head == :(=)) || throw("Invalid @graph.")

    argsyms = Expr(:vect)
    varsyms = Expr(:vect) # symbols typed as `::Var`
    for farg in expr.args[1].args
        typeof(farg) == Expr && farg.args[2] == :Var && push!(varsyms.args, farg.args[1])
        s = typeof(farg) == Expr ? farg.args[1] : farg
        push!(argsyms.args, s)
    end
    length(varsyms.args) == 0 && throw("No arguments typed `Var`.")

    body = expr.args[2].args # function body
    for s in varsyms.args
        unshift!(body, :($s.data == nothing && return Var(nothing,$varsyms,$argsyms,nothing)))
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

function compile(top::Var, inputs::Var...)
    nodes = topsort(top)
    dict = ObjectIdDict()
    syms = map(inputs) do x
        s = gensym()
        dict[x] = s
        s
    end
    for node in nodes
        length(node.args) == 0 && continue
        args = map(node.f) do arg
            typeof(arg) == Var ? dict[arg] : arg
        end
        dict[node] = Expr(:call, args...)
    end
    expr = Expr(:->, Expr(:tuple, syms...), dict[nodes[end]]) # create anonymous function
    f = eval(expr)
    Graph(nodes, f)
end

function tojson(g::Graph)
    map(g.nodes)
end
