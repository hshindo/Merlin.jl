export Var, constant, isconst

"""
    Var

`Var` is a variable type. It contains the following members:

* data
* args::Vector
* f
* df
* grad

To create an instance of `Var`, use
* Var(data)
* Var(data, grad)
"""
type Var
    data
    grad
    args::Vector
    f
    df
end

Var(data, args, f, df) = Var(data, nothing, args, f, df)
Var(data, grad) = Var(data, grad, [], nothing, nothing)
Var{T<:Real}(data::Array{T}) = Var(data, zeros(data))
Var(data::Real) = Var(data, zero(data))
Var(data) = Var(data, nothing)
Var() = Var(nothing)
constant(data) = Var(data, nothing)

Base.isconst(v::Var) = v.grad == nothing
Base.getindex(v::Var, key::Int) = v.args[key]
Base.setindex!(v::Var, value, key::Int) = v.args[key] = value

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
