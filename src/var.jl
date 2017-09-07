export Var
export zerograd
export makebatch

mutable struct Var <: AbstractVar
    data
    f
    args
    grad
    work
end

Var(data, f=nothing, args=(); grad=nothing, work=nothing) = Var(data, f, args, grad, work)

function zerograd(data::Array)
    v = Var(data)
    v.grad = zeros(data)
    v
end

function makebatch(batchsize::Int, dataset::Vector{Var}...; shf=true)
    idxs = collect(1:length(dataset[1]))
    shf && shuffle!(idxs)
    dataset = map(dataset) do vars
        vars = map(i -> vars[i], idxs)
        data = Var[]
        for i = 1:batchsize:length(vars)
            j = min(i+batchsize-1, length(vars))
            v = cat(ndims(vars[1].data), vars[i:j]...)
            v.f = nothing
            v.args = ()
            push!(data, v)
        end
        data
    end
    dataset
end
