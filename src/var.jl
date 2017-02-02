export
    Var,
    zerograd, zerograd!, setbackend!,
    topsort, gradient!

"""
    Var

`Var` is a type of variable.
It contains the following members:

* data
* f: forward function
* args: arguments of `f`
* df: backward function
* grad: gradient
"""
type Var
    data
    f
    args
    df
    grad
end

Var(data=nothing, f=nothing, args=(), df=nothing) = Var(data, f, args, df, nothing)

isvoid(x) = x == nothing
isparam(v) = isempty(v.args) && !isvoid(v.grad)

Base.getindex(v::Var, key::Int) = v.args[key]
Base.setindex!(v::Var, value, key::Int) = v.args[key] = value

function zerograd!(v::Var)
    isvoid(v.grad) && (v.grad = similar(v.data))
    fill!(v.grad, 0)
    v
end
zerograd(x) = zerograd!(Var(x))

function forward(f, args...)
    xs = map(args) do a
        isa(a, Var) ? a.data : a
    end
    y, df = forward(f, xs...)
    Var(y, f, args, df)
end

function setbackend!(v::Var, ::Type{Array})
    isa(v.data, Array) && return v
    v.data = Array(v.data)
    v.grad = isvoid(v.grad) ? v.grad : Array(v.grad)
    v
end

function topsort(top::Var)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(var::Var)
        haskey(dict, var) && return
        dict[var] = var
        for arg in var.args
            isa(arg, Var) && visit(arg)
        end
        push!(sorted, var)
    end
    visit(top)
    sorted
end

function gradient!(top::Var)
    sorted = topsort(top)
    isvoid(top.grad) && (top.grad = ones(top.data))
    for i = 1:length(sorted)
        v = sorted[i]
        isvoid(v.grad) && !isempty(v.args) && zerograd!(v)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        isvoid(v.df) && continue
        args = Any[v.grad]
        for a in v.args
            isa(a, Var) && push!(args, a.grad)
        end
        v.df(args...)
    end
    sorted
end
