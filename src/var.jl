export Var, Param

type Var
    data
    args::Vector{Var}
    f
    df
    grad
end

Var(data, args, f, df=nothing, grad=nothing) = Var(data, args, df, grad)
Var(data) = Var(data, Var[], nothing)

Param(data) = Var(data, Var[], nothing, nothing, zeros(data))

Base.size(v::Var) = size(v.data)
Base.size(v::Var, dim::Int) = size(v.data, dim)

Base.length(v::Var) = length(v.data)

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

hasgrad(v::Var) = v.grad != nothing

function topsort{T}(top::T)
    sorted = T[]
    dict = ObjectIdDict()
    function visit(v::T)
        haskey(dict,v) && return
        dict[v] = v
        for t in v.args
            typeof(t) <: T && visit(t)
        end
        push!(sorted, v)
    end
    visit(top)
    sorted
end
