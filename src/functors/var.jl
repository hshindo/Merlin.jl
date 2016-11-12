export Var, topsort, gradient!

abstract Var

Base.eltype(v::Var) = eltype(v.data)
Base.size(v::Var) = size(v.data)
Base.size(v::Var, d::Int) = size(v.data, d)
Base.getindex(v::Var, key::Int) = v.args[key]
Base.setindex!(v::Var, value, key::Int) = v.args[key] = value

constvar(data::Array) = ArrayVar(data, typeof(data)())
Var(data::Array) = ArrayVar(data)

function topsort(top::Var)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(v::Var)
        haskey(dict,v) && return
        dict[v] = v
        for a in v.args
            visit(a)
        end
        push!(sorted, v)
    end
    visit(top)
    sorted
end

function gradient!(top::Var)
    sorted = topsort(top)
    isconst(top) && (top.grad = ones(top.data))
    for i = 1:length(sorted)-1 # excludes top
        v = sorted[i]
        isconst(v) || continue
        isempty(v.args) && continue
        v.grad = zeros(v.data)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        backward!(v.f, v)
    end
    sorted
end
