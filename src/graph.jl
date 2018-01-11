export Node, Graph
export compile, getparams

mutable struct Node
end

abstract type Graph end

function getparams(g::Graph)
    params = Var[]
    for name in fieldnames(g)
        x = getfield(g, name)
        if isa(x, Var)
            push!(params, x)
        elseif applicable(getparams, x)
            append!(params, getparams(x))
        end
    end
    params
end
