export Var
export isvoid, topsort, zerograd, zerograd!, makebatch

mutable struct Var
    data
    batchdims
    args::Tuple
    df!
    grad
end

function Var(data=nothing, batchdims=nothing, args::Tuple=())
    Var(data, batchdims, args, nothing, nothing)
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

function gradient!(tops::Var...)
    sorted = topsort(tops...)
    for top in tops
        isvoid(top.grad) && (top.grad = ones(top.data))
    end
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

function makebatch(batchsize::Int, dataset::Vector{Var}...; shuffle=true)
    idxs = collect(1:length(dataset[1]))
    shuffle && shuffle!(idxs)
    map(dataset) do vars
        N = ndims(vars[1].data)
        map(1:batchsize:length(idxs)) do i
            range = i:min(i+batchsize-1,length(idxs))
            vec = map(k -> vars[k].data, range)
            if isvoid(vars[1].batchdims)
                Var(vec)
            else
                batchdata = cat(N, map(k -> vars[k].data, range)...)
                batchdims = cat(1, map(k -> vars[k].batchdims, range)...)
                Var(batchdata, batchdims)
            end
        end
    end
end

readas(::Type{Var}, x) = Var(x["data"], x["f"], x["args"])
writeas(v::Var) = Dict("data"=>v.data, "f"=>v.f, "args"=>v.args)
