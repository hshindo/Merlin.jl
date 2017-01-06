export
    Var,
    zerograd, zerograd!,
    topsort, gradient!, setbackend!

"""
    Var

`Var` is a type of variable.
It contains the following members:

* data
* f: forward or backward function
* args: arguments of `f`
* grad: gradient
"""
type Var{T}
    data::T
    f
    args
    grad
end

Var(data=nothing, f=nothing, args=()) = Var(data, f, args, nothing)
#function Var{T<:Array}(v::Var{T}, backend::Symbol)
#    backend == :cpu && return v
#    backend == :cuda && return Var(CuArray(v.data), v.f, v.args, v.grad)
#    throw("Invalid backend: $backend")
#end

Base.getindex(v::Var, key::Int) = v.args[key]
Base.setindex!(v::Var, value, key::Int) = v.args[key] = value

function zerograd!(v::Var)
    isa(v.grad, Void) && (v.grad = similar(v.data))
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
            typeof(arg) <: Var && visit(arg)
        end
        push!(sorted, var)
    end
    visit(top)
    sorted
end

function gradient!(top::Var)
    sorted = topsort(top)
    isa(top.grad, Void) && (top.grad = ones(top.data))
    for i = 1:length(sorted)
        v = sorted[i]
        isa(v.grad, Void) && !isempty(v.args) && zerograd!(v)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        isa(v.f, Void) || v.f(v.grad)
    end
    filter(v -> isempty(v.args) && !isa(v.grad,Void), sorted)
end
