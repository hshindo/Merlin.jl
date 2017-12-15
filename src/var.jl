export Var
export zerograd, isvoid, isparam, gradient!

doc"""
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

function Var(data, args=(); grad=nothing)
    Var(data, args, grad)
end
zerograd(data) = Var(data, grad=zeros(data))

Base.size(x::Var) = size(x.data)
Base.size(x::Var, i::Int) = size(x.data, i)
Base.length(x::Var) = length(x.data)
Base.ndims(x::Var) = ndims(x.data)
Base.eltype(x::Var) = eltype(x.data)
Base.strides(x::Var) = strides(x.data)
Base.stride(x::Var, i::Int) = stride(x.data, i)
isvoid(x) = x == nothing

doc"""
    isparam(x::Var)

Returns whether `x` is a parameter or not
"""
isparam(x::Var) = !isvoid(x.grad) && isempty(x.args)

doc"""
    topsort(tops::T...)

Topological sort.
"""
function topsort{T}(tops::T...)
    sorted = T[]
    dict = ObjectIdDict()
    function visit(v::T)
        haskey(dict,v) && return
        dict[v] = v
        for arg in v.args
            if isa(arg, T)
                visit(arg)
            elseif isa(arg, Vector{T})
                foreach(visit, arg)
            end
        end
        push!(sorted, v)
    end
    foreach(visit, tops)
    sorted
end

doc"""
    gradient!(top::Var)

Compute gradients.
"""
function gradient!(top::Var)
    sorted = topsort(top)
    isvoid(top.grad) && (top.grad = ones(top.data))
    for v in sorted
        if !isempty(v.args) && isvoid(v.grad)
            v.grad = zeros(v.data)
        end
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        isvoid(v.grad) && continue
        isempty(v.args) || addgrad!(v, v.args...)
    end
    filter(isparam, sorted)
end

#=
doc"""
    batch(data::Vector{Var}, batchsize::Int)
    batch(data::Vector{NTuple{N,Var}}, batchsize::Int) where N

Create batch from variables.
"""
function batch(data::Vector{Var}, batchsize::Int)
    batches = Var[]
    for i = 1:batchsize:length(data)
        T = eltype(data[i])
        N = ndims(data[i])
        batch = T[]
        batchdims = Int[]
        for k = i:min(i+batchsize-1,length(data))
            append!(batch, data[k].data)
            append!(batchdims, data[k].batchdims)
        end
        batch = reshape(batch, Base.front(size(data[i]))..., sum(batchdims))
        push!(batches, Var(batch,batchdims))
    end
    batches
end

function batch(data::Vector{NTuple{N,Var}}, batchsize::Int) where N
    res = []
    for i = 1:N
        vars = map(x -> x[i], data)
        batches = batch(vars, batchsize)
        push!(res, batches)
    end
    collect(zip(res...))
end
=#
