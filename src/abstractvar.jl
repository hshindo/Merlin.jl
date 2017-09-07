export AbstractVar
export isvoid, isparam, topsort, gradient!, zerograd!

abstract type AbstractVar end

isvoid(x) = x == nothing
isparam(x::AbstractVar) = isempty(x.args) && !isvoid(x.grad)

function zerograd!(v::AbstractVar)
    if isvoid(v.grad)
        v.grad = zeros(v.data)
    else
        fill!(v.grad, 0)
    end
    v
end

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

function addgrad!(v::AbstractVar)
    isvoid(v.f) || addgrad!(v, v.f, v.args...)
end

function gradient!(top::AbstractVar)
    sorted = topsort(top)
    isvoid(top.grad) && (top.grad = ones(top.data))
    for v in sorted
        !isempty(v.args) && isvoid(v.grad) && zerograd!(v)
    end
    for i = length(sorted):-1:1
        addgrad!(sorted[i])
    end
    sorted
end
