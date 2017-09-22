export Var
export zerograd, batchsize, data

mutable struct Var <: AbstractVar
    data
    batchdims
    f
    args
    grad
    work
end

function Var(data, batchdims=nothing, f=nothing, args=(); hasgrad=false, work=nothing)
    batchdims == nothing && (batchdims = [size(data)[end]])
    grad = hasgrad ? zeros(data) : nothing
    Var(data, batchdims, f, args, grad, work)
end

Base.size(x::Var) = size(x.data)
Base.size(x::Var, i::Int) = size(x.data, i)
Base.length(x::Var) = length(x.data)
Base.ndims(x::Var) = ndims(x.data)
Base.eltype(x::Var) = eltype(x.data)

batchsize(x::Var) = x.batchdims
data(x::Var) = x.data

update!(x::Var, opt) = opt(x.data, x.grad)
