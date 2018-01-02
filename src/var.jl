export Var
export zerograd, isvoid, isparam, gradient!, setbackend!

doc"""
    Var

Variable struct.

`Var` contains the following members:
* data
* args
* grad
* work

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
    work
end

Var(data=nothing, args=(); grad=nothing, work=()) = Var(data, args, grad, work)
zerograd(data) = Var(data, grad=zeros(data))

Base.size(x::Var) = size(x.data)
Base.size(x::Var, i::Int) = size(x.data, i)
Base.length(x::Var) = length(x.data)
Base.ndims(x::Var) = ndims(x.data)
Base.eltype(x::Var) = eltype(x.data)
Base.strides(x::Var) = strides(x.data)
Base.stride(x::Var, i::Int) = stride(x.data, i)
Base.getindex(x::Var, i::Int) = x.args[i]
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
function topsort(tops::T...) where T
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
function gradient!(tops::Var...)
    sorted = topsort(tops...)
    for top in tops
        isvoid(top.grad) && (top.grad = ones(top.data))
    end
    for v in sorted
        if !isempty(v.args) && isvoid(v.grad)
            v.grad = zeros(v.data)
        end
    end
    for i = length(sorted):-1:1
        y = sorted[i]
        isvoid(y.grad) && continue
        isempty(y.args) && continue
        addgrad!(y, y.args...)
    end
    filter(isparam, sorted)
end

doc"""
    setbackend!(x::Var, backend::String)

* backend: "CPU" or "CUDA"
"""
function setbackend!(x::Var, backend::String)
    if backend == "CPU"
        if isa(x.data, CuArray)
            x.data = Array(x.data)
            isvoid(x.grad) || (x.grad = Array(x.grad))
        end
    elseif startswith(backend, "CUDA")
        if isa(x.data, Array)
            dev = parse(Int, backend[6:end])
            LibCUDA.setdevice(dev)
            x.data = CuArray(x.data)
            isvoid(x.grad) || (x.grad = CuArray(x.grad))
        end
    else
        throw("Unknown backend: $backend")
    end
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
