export Var
export parameter, parameters, zerograd!, isnothing, isparam

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

"""
    isparam(x::Var)

Returns whether `x` is a parameter or not
"""
isparam(x) = isa(x,Var) && !isnothing(x.grad) && isempty(x.args)

function parameters(xs...)
    vars = Var[]
    for x in xs
        if isa(x, Var)
            push!(vars, x)
        elseif isa(x, Tuple)
            append!(vars, x)
        else
            throw("Invalid parameter: $x")
        end
    end
    filter(isparam, vars)
end
#parameters(x::Var) = (x,)
#parameters(x::Tuple{Vararg{Var,N}}) = x
#parameters(x::Parametric) = parameters(x)
