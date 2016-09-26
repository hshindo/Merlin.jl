export Var, constant, isconst

"""
    Var

`Var` is a variable type. It contains the following members:

* data
* grad
* args::Vector{Var}
* f
* df

To create an instance of `Var`, use
* Var(data)
* Var(data, grad)
"""
type Var
    data
    grad
    args::Vector{Var}
    f
    df
end

Var(data, args, f, df) = Var(data, nothing, args, f, df)
Var(data, grad) = Var(data, grad, Var[], nothing, nothing)
Var(data) = Var(data, zeros(data))
Var() = Var(nothing, nothing)

hasgrad(v::Var) = v.grad != nothing

constant(data) = Var(data, nothing)
Base.isconst(v::Var) = v.grad == nothing
dataof(v::Var) = v.data

Base.getindex(v::Var, key::Int) = v.args[key]

function topsort(top::Var)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(var::Var)
        haskey(dict, var) && return
        dict[var] = var
        foreach(visit, var.args)
        push!(sorted, var)
    end
    visit(top)
    sorted
end

h5convert(x::Var) = h5convert(x.data)
h5load!(::Type{Var}, data::Dict) = Var(data["data"])
