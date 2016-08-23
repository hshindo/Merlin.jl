abstract AbstractNode

Base.getindex(n::AbstractNode, key) = n.args[key]
Base.setindex!(n::AbstractNode, value, key) = n.args[key] = value

Base.length(n::AbstractNode) = length(n.args)

function topsort{T}(top::T)
    sorted = T[]
    dict = ObjectIdDict()
    function visit(node::AbstractNode)
        haskey(dict,node) && return
        dict[node] = node
        for a in node.args
            typeof(a) == T && visit(a)
        end
        push!(sorted, node)
    end
    visit(top)
    sorted
end

function bottomup{T}(f, node::T)
    for arg in node.args
        typeof(arg) == T && bottomup(f, arg)
    end
    f(node)
end
