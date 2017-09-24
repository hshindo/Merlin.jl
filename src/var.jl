export Var
export zerograd, batchsize, isvoid, topsort, gradient!, update!

mutable struct Var
    data
    batchdims
    f
    args
    grad
end

function Var(data, batchdims=nothing, f=nothing, args=(); fixed=true)
    batchdims == nothing && (batchdims = [size(data,ndims(data))])
    grad = fixed ? nothing : zeros(data)
    Var(data, batchdims, f, args, grad)
end

Base.size(x::Var) = size(x.data)
Base.size(x::Var, i::Int) = size(x.data, i)
Base.length(x::Var) = length(x.data)
Base.ndims(x::Var) = ndims(x.data)
Base.eltype(x::Var) = eltype(x.data)

batchsize(x::Var) = x.batchdims
batchsize(x::Var, i::Int) = x.batchdims[i]
isvoid(x) = x == nothing

update!(x::Var, opt) = opt(x.data, x.grad)

function topsort{T}(top::T)
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
    visit(top)
    sorted
end

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
        isvoid(v.f) || addgrad!(v, v.f, v.args...)
    end
    filter(sorted) do v
        isempty(v.args) && !isvoid(v.grad)
    end
end
