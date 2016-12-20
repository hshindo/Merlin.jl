export
    Var,
    zerograd, zerograd!,
    topsort, gradient!, setbackend!

"""
    Var

`Var` is a type of variable.

```julia
Var(data, [args=()])
```
"""
type Var
    data
    f
    args
    grad
end

Var(data=nothing, f=nothing, args=()) = Var(data, f, args, nothing)

Base.getindex(v::Var, key::Int) = v.args[key]
Base.setindex!(v::Var, value, key::Int) = v.args[key] = value
Base.eltype(v::Var) = eltype(v.data)
Base.size(v::Var) = size(v.data)
Base.size(v::Var, dim::Int) = size(v.data, dim)
Base.length(v::Var) = length(v.data)
Base.ndims(v::Var) = ndims(v.data)

function zerograd!(v::Var)
    v.grad == nothing && (v.grad = similar(v.data))
    fill!(v.grad, 0)
    v
end
zerograd(x) = zerograd!(Var(x))

function setbackend!{T}(v::Var, ::Type{T})
    typeof(v.data) <: T && return v
    v.data = T(v.data)
    v.grad == nothing || (v.grad = T(v.grad))
end

function topsort(top::Var)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(var::Var)
        haskey(dict, var) && return
        dict[var] = var
        for arg in var.args
            typeof(arg) <: Var && visit(arg)
        end
        push!(sorted, var)
    end
    visit(top)
    sorted
end

function gradient!(top::Var)
    sorted = topsort(top)
    top.grad == nothing && (top.grad = ones(top.data))
    for i = 1:length(sorted)
        v = sorted[i]
        v.grad == nothing && !isempty(v.args) && zerograd!(v)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        v.f == nothing || v.f(v.grad)
    end
    sorted
end
