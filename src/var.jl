export Var
export isvoid, topsort, zerograd, zerograd!

type Var
    data
    f
    args::Tuple
    df!
    grad
end

Var(data, f=nothing, args=()) = Var(data, f, args, nothing, nothing)
zerograd(data) = Var(data, nothing, (), nothing, zeros(data))

isvoid(x) = x == nothing
isconst(x::Var) = x.grad == nothing
Base.getindex(x::Var, key::Int) = x.args[key]
getdata(x::Var) = x.data
getgrad(x::Var) = x.grad

function zerograd!(v::Var)
    if isvoid(v.grad)
        v.grad = zeros(v.data)
    else
        fill!(v.grad, 0)
    end
    v
end

function topsort(top::Var)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(v::Var)
        haskey(dict,v) && return
        dict[v] = v
        for arg in v.args
            if isa(arg, Var)
                visit(arg)
            elseif isa(arg, Vector{Var})
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
    isconst(top) && (top.grad = ones(top.data))
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
