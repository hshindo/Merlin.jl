export Var, constant, isconst, topsort, gradient!

"""
    Var

`Var` is a type of variable. `Var` holds forward/backward information.
It contains the following members:

* data: data value
* grad: gradient value
* args::Vector: arguments
* df: diff function

To create an instance of `Var`, use
* Var(data)
* Var(data, grad)
"""
type Var{T}
    data::T
    grad
    f
    args::Vector
    df

    Var(data::T, grad) = new(data, grad, Var[])
    Var(data::T, grad, args::Vector, df::Function) = new(data, grad, args, df)
end

Var{T}(data::T) = Var{T}(data, zeros(data))
Var{T}(data::T, args, df) = Var{T}(data, zeros(T), args, df)

constant{T}(data::T) = Var(data, T())

Base.isconst(v::Var) = isempty(v.grad)
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
        isdefined(v, :df) && v.df(v.grad)
        #v.df == nothing || v.df(v.grad)
    end
    sorted
end
