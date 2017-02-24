export
    Var,
    zerograd, zerograd!, isvoid,
    topsort, gradient!

"""
    Var

`Var` is a type of variable.
It contains the following members:

* data
* f: forward function
* args: arguments
* df: backward function
* grad: gradient
"""
type Var
    data
    args
    df
    grad
end

Var(data, args=(), df=nothing) = Var(data, args, df, nothing)

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

macro forward(expr)
    args = map(a -> isa(a,Expr) ? a.args[1] : a, expr.args)
    Expr(:(=), expr, Expr(:call, :forward0, args...))
end

function forward0(args...)
    xs = map(args) do arg
        isa(arg, Var) && return arg.data
        isa(arg, Vector{Var}) && return map(a -> a.data, arg)
        arg
    end
    y, backward! = forward(xs...)
    Var(y, args, backward!)
end

function setbackend!{T<:Array}(v::Var, ::Type{T})
    isa(v.data, Array) && return v
    v.data = Array(v.data)
    v.grad = isvoid(v.grad) ? v.grad : Array(v.grad)
    v
end

function setbackend!{T<:CuArray}(v::Var, ::Type{T})
    isa(v.data, CuArray) && return v
    v.data = CuArray(v.data)
    v.grad = isvoid(v.grad) ? v.grad : CuArray(v.grad)
    v
end

function topsort(top::Var)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(var::Var)
        haskey(dict, var) && return
        dict[var] = var
        for arg in var.args
            if isa(arg, Var)
                visit(arg)
            elseif isa(arg, Vector{Var})
                foreach(visit, arg)
            end
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
        for arg in v.args
            isa(arg, Var) && push!(args, arg.grad)
            isa(arg, Vector{Var}) && push!(args, map(a -> a.grad, arg))
        end
        v.df(args...)
    end
    sorted
end
