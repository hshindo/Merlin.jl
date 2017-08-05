export Var
export isvoid, topsort, gradient!, zerograd, zerograd!
export makebatch

mutable struct Var{T}
    data::T
    batchdims
    f
    args
    grad
    work
end

function Var(data::T, batchdims=nothing, f=nothing, args=(); grad=nothing, work=nothing) where {T}
    if isvoid(batchdims) && T <: Array
        batchdims = [size(data,ndims(data))]
    end
    Var{T}(data, batchdims, f, args, grad, work)
end

isvoid(x) = x == nothing
# Base.getindex(x::Var, key::Int) = x.args[key]

function zerograd(data)
    v = Var(data)
    v.grad = zeros(data)
    v
end

function zerograd!(v::Var)
    if isvoid(v.grad)
        v.grad = zeros(v.data)
    else
        fill!(v.grad, 0)
    end
    v
end

function topsort(::Type{T}, tops::T...) where {T}
    sorted = T[]
    dict = ObjectIdDict()
    function visit(v::T)
        haskey(dict,v) && return
        dict[v] = v
        for arg in v.args
            isa(arg,T) && visit(arg)
        end
        push!(sorted, v)
    end
    foreach(visit, tops)
    sorted
end

addgrad!(v::Var) = addgrad!(v, v.f, v.args...)

function gradient!(tops::Var...)
    sorted = topsort(Var, tops...)
    for top in tops
        isvoid(top.grad) && (top.grad = ones(top.data))
    end
    for i = 1:length(sorted)
        v = sorted[i]
        !isempty(v.args) && isvoid(v.grad) && zerograd!(v)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        isvoid(v.f) || addgrad!(v)
    end
    sorted
end

function makebatch(batchsize::Int, dataset::Vector{<:Var}...)
    idxs = collect(1:length(dataset[1]))
    shuffle!(idxs)
    dataset = map(dataset) do vars
        map(1:batchsize:length(idxs)) do i
            vs = map(k -> vars[idxs[k]], i:min(i+batchsize-1,length(idxs)))
            v = cat(ndims(vs[1].data), vs...)
            v.f = nothing
            v.args = ()
            v
        end
    end
    dataset
end
