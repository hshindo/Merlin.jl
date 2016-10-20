export Var, constant, isconst, gradient!

"""
    Var

`Var` is a type of variable. `Var` holds forward/backward information.
It contains the following members:

* data: data value
* grad: gradient value
* args::Vector: arguments
* f: function
* df: diff function

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

zerograd!(v::Var) = 

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

function gradient!(top::Var)
    sorted = topsort(top)
    isconst(top) && (top.grad = ones(top.data))
    for i = 1:length(sorted)-1 # excludes top
        v = sorted[i]
        (!isconst(v) || isempty(v.args)) && continue
        v.grad = zeros(v.data)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        v.df == nothing || v.df(v.grad)
    end
    sorted
end
