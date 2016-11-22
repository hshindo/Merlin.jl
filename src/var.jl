export Var, constant, isconst, topsort, gradient!, zerograd, zerograd!

"""
    Var

`Var` is a type of variable.

```julia
Var(data, [args=()])
```
"""
type Var
    data
    args
    grad
    df
end

Var(data, args=()) = Var(data, args, nothing, nothing)
Var(T::Type, dims::Tuple, args=()) = Var(alloc(T,dims), args)
Var() = Var(nothing)

Base.isconst(v::Var) = v.grad == nothing
Base.getindex(v::Var, key::Int) = v.args[key]
Base.setindex!(v::Var, value, key::Int) = v.args[key] = value
Base.eltype(v::Var) = eltype(v.data)
Base.size(v::Var) = size(v.data)
Base.size(v::Var, dim::Int) = size(v.data, dim)
Base.length(v::Var) = length(v.data)
Base.ndims(v::Var) = ndims(v.data)
Base.similar(v::Var, args=()) = Var(eltype(v), size(v), args)

function zerograd!(v::Var)
    v.grad == nothing && (v.grad = alloc(eltype(v),size(v)))
    fill!(v.grad, 0)
    v
end
zerograd(x) = zerograd!(Var(x))

function topsort(top::Var)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(var::Var)
        haskey(dict, var) && return
        dict[var] = var
        for arg in var.args
            typeof(arg) == Var && visit(arg)
        end
        push!(sorted, var)
    end
    visit(top)
    sorted
end

function gradient!(top::Var)
    sorted = topsort(top)
    isconst(top) && (top.grad = ones(top.data))
    for i = 1:length(sorted)
        v = sorted[i]
        isconst(v) && !isempty(v.args) && zerograd!(v)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        v.df == nothing || v.df()
    end
    sorted
end
