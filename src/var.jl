export Var
export zerograd
export makebatch

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

function makebatch(batchsize::Int, dataset::Vector{Var}...)
    idxs = collect(1:length(dataset[1]))
    shuffle!(idxs)
    dataset = map(dataset) do vars
        map(1:batchsize:length(idxs)) do i
            vs = map(k -> vars[idxs[k]], i:min(i+batchsize-1,length(idxs)))
            v = cat(ndims(vs[1].data), vs...)
            v.f = nothing
            v.args = ()
            v
        end
    end
    dataset
end
