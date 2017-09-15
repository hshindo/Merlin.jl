export Var
export zerograd, data

mutable struct Var <: AbstractVar
    data
    batchdims
    f
    args
    grad
    work
end

function Var(data, batchdims=nothing, f=nothing, args=(); grad=nothing, work=nothing)
    if isvoid(batchdims) && isa(data,Array)
        batchdims = [size(data,ndims(data))]
    end
    Var(data, batchdims, f, args, grad, work)
end

Base.size(x::Var) = size(x.data)
Base.size(x::Var, i::Int) = size(x.data, i)
Base.length(x::Var) = length(x.data)
data(x::Var) = x.data

function zerograd(data)
    v = Var(data)
    v.grad = zeros(data)
    v
end
