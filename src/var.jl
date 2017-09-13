export Var
export zerograd, batchsize

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

function zerograd(data::Array)
    v = Var(data)
    v.grad = zeros(data)
    v
end
