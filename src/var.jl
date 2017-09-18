export Var
export zerograd

mutable struct Var <: AbstractVar
    data
    sizes
    f
    args
    grad
    work
end

function Var(data, sizes=nothing, f=nothing, args=(); grad=nothing, work=nothing)
    Var(data, sizes, f, args, grad, work)
end

Base.size(x::Var) = size(x.data)
Base.size(x::Var, i::Int) = size(x.data, i)
Base.length(x::Var) = length(x.data)
Base.ndims(x::Var) = ndims(x.data)

isparam(x::Var) = isempty(x.args) && length(x.batchdims) == 1

function zerograd(data)
    v = Var(data)
    v.grad = zeros(data)
    v
end
