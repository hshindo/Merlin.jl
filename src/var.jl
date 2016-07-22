export Var, Param

type Var
  data
  args::Vector{Var}
  df
  grad
end

Var(data, args=Var[], df=nothing) = Var(data, args, df, nothing)
Param(data) = Var(data, Var[], nothing, zeros(data))

Base.size(v::Var) = size(v.data)
Base.size(v::Var, dim::Int) = size(v.data, dim)

Base.getindex(v::Var, key) = v.args[key]
Base.setindex!(v::Var, value, key) = v.args[key] = value

hasgrad(v::Var) = v.grad != nothing

function topsort(top::Var)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(v)
        haskey(dict, v) && return
        dict[v] = v
        for t in v.args
            visit(t)
        end
        push!(sorted, v)
    end
    visit(top)
    sorted
end
