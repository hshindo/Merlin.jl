export Var
export param, zerograd!, batchsize, isvoid, isparam, gradient!, topsort, create_batch

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
    args
    grad
end

Var(data, args=()) = Var(data, args, nothing)

function param(data)
    v = Var(data)
    v.grad = fill!(similar(data), 0)
    v
end

function zerograd!(x::Var)
    isvoid(x.grad) && throw("")
    fill!(x.grad, 0)
    x
end

Base.size(x::Var) = size(x.data)
Base.size(x::Var, i::Int) = size(x.data, i)
Base.strides(x::Var) = strides(x.data)
Base.stride(x::Var, i::Int) = stride(x.data, i)
Base.length(x::Var) = length(x.data)
Base.ndims(x::Var) = ndims(x.data)
Base.eltype(x::Var) = eltype(x.data)

isvoid(x) = x == nothing
oncpu(x::Var) = isa(x.data, Array)
oncuda(x::Var) = isa(x.data, CuAray)

"""
    isparam(x::Var)

Returns whether `x` is a parameter or not
"""
isparam(x) = isa(x,Var) && !isvoid(x.grad) && isempty(x.args)

"""
    topsort(tops::T...)

Topological sort.
"""
function topsort(tops::T...) where T
    sorted = T[]
    dict = IdDict()
    function visit(v::T)
        haskey(dict,v) && return
        dict[v] = v
        for arg in v.args
            if isa(arg, T)
                visit(arg)
            #elseif isa(arg, Vector{T})
            #    foreach(visit, arg)
            end
        end
        push!(sorted, v)
    end
    foreach(visit, tops)
    sorted
end

"""
    gradient!(top::Var)

Compute gradients.
"""
function gradient!(tops::Var...)
    sorted = topsort(tops...)
    for top in tops
        if isvoid(top.grad)
            top.grad = fill!(similar(top.data), 1)
        end
    end
    for v in sorted
        if !isempty(v.args) && isvoid(v.grad)
            v.grad = fill!(similar(v.data), 0)
        end
    end
    for i = length(sorted):-1:1
        y = sorted[i]
        all(y.args) do x
            isa(x,Var) &&
        end
        isvoid(y.grad) && continue
        isempty(y.args) && continue
        addgrad!(y, y.args...)
    end
    collect(Iterators.filter(isparam,sorted))
end

function configure!(xs::Var...)
    if iscpu()
        f = tocpu
    elseif iscuda()
        f = tocuda
    end
    for x in xs
        x.data = f(x.data)
        isvoid(x.grad) || (x.grad = f(x.grad))
    end
end
tocpu(x::Array) = x
tocpu(x::CuArray) = Array(x)
tocpu(x::CuArray{Cint}) = Array{Int}(Array(x))
tocuda(x::CuArray) = x
tocuda(x::Array{Int}) = CuArray(Array{Cint}(x))
tocuda(x::Array) = CuArray(x)

function create_batch(f::Function, batchsize::Int, samples::Vector{T}) where T
    batches = []
    for i = 1:batchsize:length(samples)
        range = i:min(i+batchsize-1,length(samples))
        batch = f(samples[range])
        push!(batches, batch)
    end
    batches
end
