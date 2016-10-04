export Var, constant, isconst

"""
    Var

`Var` is a variable type. It contains the following members:

* data
* args::Vector
* df
* grad

To create an instance of `Var`, use
* Var(data)
* Var(data, grad)
"""
type Var
    data
    args::Vector
    df
    grad
end

Var(data, args, df) = Var(data, args, df, nothing)
Var(data, grad) = Var(data, Var[], nothing, grad)
Var(data::Array) = Var(data, zeros(data))
Var(data::Number) = Var(data, zero(data))
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

h5object(x::Var) = Dict("data"=>x.data, "args"=>x.args)
h5load(::Type{Var}, x::Dict) = Var(x["data"], x["args"], nothing)
