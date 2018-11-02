export Var
export parameter, zerograd!, isnothing, isparam, gradient!, topsort

"""
    Var

Variable struct.

`Var` contains the following members:
* data
* args
* grad

# Example
```julia
T = Float32
x = Var(rand(T,10,5)) # x.grad is set to `nothing`
x = zerograd(rand(T,10,5)) # x.grad is initialized as zero.
```
"""
mutable struct Var
    data
    f
    args
    grad
end

Var(data, f=nothing, args=()) = Var(data, f, args, nothing)

function parameter(data)
    v = Var(data)
    v.grad = fill!(similar(v.data), 0)
    v
end

function zerograd!(x::Var)
    fill!(x.grad, 0)
    x
end

data(x::Var) = x.data
gradient(x::Var) = x.grad
Base.size(x::Var) = size(x.data)
Base.size(x::Var, i::Int) = size(x.data, i)
Base.length(x::Var) = length(x.data)
Base.ndims(x::Var) = ndims(x.data)
Base.eltype(x::Var) = eltype(x.data)
isnothing(x) = x == nothing

#getdevice(x::Var) = getdevice(x.data)
#getdevice(x::Array) = -1
#getdevice(x::CuArray) = CUDA.getdevice(x)

"""
    isparam(x::Var)

Returns whether `x` is a parameter or not
"""
isparam(x) = isa(x,Var) && !isnothing(x.grad) && isempty(x.args)

"""
    topsort(tops::T...)

Topological sort.
"""
function topsort(top::T) where T
    sorted = T[]
    dict = IdDict{T,T}()
    function visit(x::T)
        haskey(dict,x) && return
        dict[x] = x
        for arg in x.args
            isa(arg,T) && visit(arg)
        end
        push!(sorted, x)
    end
    visit(top)
    sorted
end

"""
    gradient!(top::Var)

Compute gradients.
"""
function gradient!(top::Var)
    sorted = topsort(top)
    if isnothing(top.grad)
        top.grad = fill!(similar(top.data), 1)
    end
    for v in sorted
        if !isempty(v.args) && isnothing(v.grad)
            v.grad = fill!(similar(v.data), 0)
        end
    end
    for i = length(sorted):-1:1
        y = sorted[i]
        isnothing(y.grad) && continue
        isempty(y.args) && continue
        y.f(y, y.args...)
    end
    sorted
end

function tocpu!(x::Var)
end

function tocuda!(x::Var)
end

function configure2!(xs::Vararg{Var})
    return

    if iscpu()
        f = tocpu
    elseif iscuda()
        f = tocuda
    end
    for x in xs
        x.data = f(x.data)
        isnothing(x.grad) || (x.grad = f(x.grad))
    end
end
