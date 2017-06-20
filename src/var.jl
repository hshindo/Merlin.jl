export Var
export isvoid, topsort, zerograd, zerograd!

mutable struct Var
    data
    batchdims
    f
    args::Tuple
    df!
    grad
end

function Var(data, batchdims=nothing, f=nothing, args=())
    Var(data, batchdims, f, args, nothing, nothing)
end

function Var(data::Vector{Array{T,N}}) where {T,N}
    batchdims = cat(1, map(length,data)...)
    data = cat(N, data...)
    Var(data, batchdims)
end

function zerograd(data)
    v = Var(data)
    v.grad = zeros(data)
    v
end

isvoid(x) = x == nothing
Base.getindex(x::Var, key::Int) = x.args[key]

function zerograd!(v::Var)
    if isvoid(v.grad)
        v.grad = zeros(v.data)
    else
        fill!(v.grad, 0)
    end
    v
end

function topsort(tops::Var...)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(v::Var)
        haskey(dict,v) && return
        dict[v] = v
        for arg in v.args
            isa(arg,Var) && visit(arg)
        end
        push!(sorted, v)
    end
    foreach(visit, tops)
    sorted
end

function gradient!(top::Var)
    sorted = topsort(top)
    isvoid(top.grad) && (top.grad = ones(top.data))
    for i = 1:length(sorted)
        v = sorted[i]
        isempty(v.args) && continue
        isvoid(v.grad) && zerograd!(v)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        isvoid(v.df!) || v.df!()
    end
    sorted
end

readas(::Type{Var}, x) = Var(x["data"], x["f"], x["args"])
writeas(v::Var) = Dict("data"=>v.data, "f"=>v.f, "args"=>v.args)
