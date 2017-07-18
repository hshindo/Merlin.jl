export Var
export isvoid, topsort, zerograd, zerograd!

mutable struct Var
    data
    f
    args
    df!
    grad
end

function Var(data=nothing, f=nothing, args=())
    Var(data, f, args, nothing, nothing)
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

function forward(f, args...)
    y = Var(nothing, f, args)
    any(a -> isvoid(a.data), args) && return y
    f(y, args...)
    y
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
        !isempty(v.args) && isvoid(v.grad) && zerograd!(v)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        isvoid(v.df!) || v.df!()
    end
    sorted
end

function makebatch(batchsize::Int, dataset::Vector{Var}...)
    idxs = collect(1:length(dataset[1]))
    shuffle!(idxs)
    dataset = map(dataset) do vars
        N = ndims(vars[1].data)
        map(1:batchsize:length(idxs)) do i
            range = map(k -> idxs[k], i:min(i+batchsize-1,length(idxs)))
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
    dataset
end

writeas(v::Var) = Dict("data"=>v.data, "f"=>v.f, "args"=>v.args)
readas(::Type{Var}, d::Dict) = Var(d["data"], d["f"], d["args"])

JLD2.writeas(::Type{Var}) = Dict{Any,Any}
function JLD2.wconvert(::Type{Dict{Any,Any}}, v::Var)
    dict = Dict()
    dict["data"] = v.data
    dict["f"] = v.f
    dict["args"] = v.args
    dict
end
function JLD2.rconvert(::Type{Var}, dict::Dict{Any,Any})
    Var(dict["data"], dict["f"], dict["args"])
end
