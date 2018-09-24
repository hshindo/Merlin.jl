export Var
export param, zerograd!, isnothing, isparam, gradient!, topsort, create_batch

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
    name::String
end

Var(data, f=nothing, args=(); name="") = Var(data, f, args, nothing, name)

function param(data)
    v = Var(data)
    v.grad = fill!(similar(data), 0)
    v
end

zerograd!(x::Var) = fill!(x.grad, 0)

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
isparam(x) = isa(x,Var) && !isa(x.grad,Nothing) && isempty(x.args)

"""
    topsort(tops::T...)

Topological sort.
"""
function topsort(top::Var)
    sorted = Var[]
    dict = IdDict{Var,Var}()
    function visit(v::Var)
        haskey(dict,v) && return
        dict[v] = v
        for arg in v.args
            if arg isa Var
                visit(arg)
            #elseif isa(arg, Vector{T})
            #    foreach(visit, arg)
            end
        end
        push!(sorted, v)
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
        if !isempty(v.args) && v.grad isa Nothing
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

function configure!(xs::Vararg{Var})
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
tocpu(x::Array) = x
tocpu(x::CuArray) = Array(x)
tocpu(x::CuArray{Cint}) = Array{Int}(Array(x))
tocuda(x::CuArray) = x
tocuda(x::Array{Int}) = CuArray(Array{Cint}(x))
tocuda(x::Array) = CuArray(x)

function create_batch(batchsize::Int, samples::Vector)
    batches = []
    for i = 1:batchsize:length(samples)
        range = i:min(i+batchsize-1,length(samples))
        push!(batches, samples[range])
    end
    batches
end
