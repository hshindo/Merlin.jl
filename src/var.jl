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
    f
    args
    df!
    grad
end

Var(data=nothing, f=nothing, args=(), (df!)=nothing) = Var(data, f, args, df!, nothing)

isvoid(x) = x == nothing
isparam(v::Var) = isempty(v.args) && !isvoid(v.grad)

Base.getindex(v::Var, key::Int) = v.args[key]
Base.setindex!(v::Var, value, key::Int) = v.args[key] = value

function zerograd!(v::Var)
    isvoid(v.grad) && (v.grad = similar(v.data))
    fill!(v.grad, 0)
    v
end
zerograd(x) = zerograd!(Var(x))

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

function topsort(top::Var...)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(var::Var)
        haskey(dict,var) && return
        dict[var] = var
        foreach(visit, var.args)
        push!(sorted, var)
    end
    foreach(visit, top)
    sorted
end

function gradient!(top::Var...)
    sorted = topsort(top...)
    for v in top
        isvoid(v.grad) && (v.grad = ones(v.data))
    end
    for i = 1:length(sorted)
        v = sorted[i]
        isvoid(v.grad) && !isempty(v.args) && zerograd!(v)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        isvoid(v.df!) || v.df!()
        #gs = map(a -> a.grad, v.args)
        #v.df!(v.grad, gs...)
    end
    sorted
end

readas(::Type{Var}, x) = Var(x["data"], x["f"], x["args"])
writeas(v::Var) = Dict("data"=>v.data, "f"=>v.f, "args"=>v.args)
