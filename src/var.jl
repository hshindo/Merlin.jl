export Var
export zerograd, batchsize, data, isvoid, topsort, gradient!, update!

mutable struct Var
    data
    batchdims
    f
    args
    grad
end

function Var(data, batchdims=nothing, f=nothing, args=(); hasgrad=false)
    batchdims == nothing && (batchdims = [size(data)[end]])
    grad = hasgrad ? zeros(data) : nothing
    Var(data, batchdims, f, args, grad)
end

Base.size(x::Var) = size(x.data)
Base.size(x::Var, i::Int) = size(x.data, i)
Base.length(x::Var) = length(x.data)
Base.ndims(x::Var) = ndims(x.data)
Base.eltype(x::Var) = eltype(x.data)

batchsize(x::Var) = x.batchdims
data(x::Var) = x.data
isvoid(x) = x == nothing

update!(x::Var, opt) = isempty(x.args) && !isvoid(x.grad) && opt(x.data,x.grad)

function topsort{T}(top::T)
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
    visit(top)
    sorted
end

function update!(vars::Vector{Var}, opt)
    for v in vars
        # isa(v,Var) || continue
        isempty(v.args) && !isvoid(v.grad) && opt(v.data,v.grad)
        #for arg in v.args
        #    applicable(update!,arg,opt) && update!(arg,opt)
        #end
    end
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
    sorted
end
