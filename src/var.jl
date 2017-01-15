export
    Var,
    zerograd, zerograd!, setbackend,
    topsort, gradient!

"""
    Var

`Var` is a type of variable.
It contains the following members:

* data
* f: forward or backward function
* args: arguments of `f`
* grad: gradient
"""
type Var
    data
    f
    args
    grad
end

Var(data=nothing, f=nothing, args=()) = Var(data, f, args, nothing)

isvoid(x) = x == nothing
#=
function Var{T<:Array}(v::Var{T}, backend::Symbol)
    backend == :cpu && return v
    if backend == :cuda
        grad = isa(v.grad, Void) ? nothing : CudaArray(v.grad)
        return Var(CudaArray(v.data), v.f, v.args, grad)
    end
    throw("Invalid backend: $backend")
end
=#

Base.getindex(v::Var, key::Int) = v.args[key]
Base.setindex!(v::Var, value, key::Int) = v.args[key] = value

function zerograd!(v::Var)
    isvoid(v.grad) && (v.grad = similar(v.data))
    fill!(v.grad, 0)
    v
end
zerograd(x) = zerograd!(Var(x))

function forward(f::Function, args...)::Var
    any(a -> isa(a, Var) && isvoid(a.data), args) && return Var(nothing, f, args)
    xs = map(args) do a
        isa(a, Var) ? a.data : a
    end
    y, df = forward(f, xs...)
    Var(y, df, args)
end

#=
function setbackend{T1<:Array,T2}(v::Var{T1}, ::Type{T2})
    T2 <: Array && return v
    if T2 <: CudaArray
        grad = isa(v.grad, Void) ? v.grad : CudaArray(v.grad)
        return Var(CudaArray(v.data), v.f, v.args, grad)
    end
    throw("Invalid type: $T2")
end
=#

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
        isvoid(v.f) && continue
        args = map(v.args) do a
            isa(a, Var) ? a.grad : a
        end
        v.f(v.grad, args...)
    end
    filter(v -> isempty(v.args) && !isvoid(v.grad), sorted)
end
