type Var
    data
    args
    df!
    grad
end

Var(data, args=(), df!=nothing, grad=nothing) = Var(data, args, df!, grad)

isvoid(x) = x == nothing

Base.getindex(v::Var, key::Int) = v.args[key]

constant(data) = Var(data)
param(data) = Var(data, (), nothing)

function forward!(top::Var)
    vars = topsort(top)
    

end

function topsort(tops::Var...)
    sorted = Var[]
    dict = ObjectIdDict()
    function visit(var::Var)
        haskey(dict,var) && return
        dict[var] = var
        for arg in var.args
            isa(arg,Var) && visit(arg)
        end
        push!(sorted, var)
    end
    foreach(visit, tops)
    sorted
end

function gradient!(tops::Var...)
    sorted = topsort(tops...)
    for v in tops
        isvoid(v.grad) && (v.grad = ones(v.data))
    end
    for i = 1:length(sorted)
        v = sorted[i]
        isempty(v.args) && continue
        #all(a -> isvoid(a.grad), v.args) && continue
        isvoid(v.grad) && zerograd!(v)
    end
    for i = length(sorted):-1:1
        v = sorted[i]
        isvoid(v.df!) || v.df!()
    end
    sorted
end
