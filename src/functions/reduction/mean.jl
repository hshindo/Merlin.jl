import Base.mean

"""
    mean(x, dim::Int)

Computes the average over the given dimension.
"""
function mean(x::Var, dim::Int)
    if dim == ndims(x)
        y = mean_batch(x.data, x.batchdims)
        batchdims = ones(Int, length(x.batchdims))
    else
        y = mean(x.data, dim)
        batchdims = x.batchdims
    end
    Var(y, batchdims, mean, (x,dim))
end

mean(x::Node, dim::Int; name="") = Node(mean, (x,dim), name)

function mean_batch{T,N}(x::Array{T,N}, batchdims::Vector{Int})
    front = Base.front(size(x))
    n = prod(front)
    y = T[]

    cumdim = 0
    for i = 1:length(batchdims)
        p = pointer(x, n*cumdim+1)
        subx = unsafe_wrap(Array, p, (front...,batchdims[i]))

        m = mean(subx, N)
        append!(y, m)
        cumdim += batchdims[i]
    end
    reshape(y, front..., length(batchdims))
end

function addgrad!(y::Var, ::typeof(mean), x::Var, dim::Int)
    isvoid(x.grad) || ∇mean!(y.grad, x.grad, dim)
end

function ∇mean!{T}(gy::Array{T}, gx::Array{T}, dim::Int)
    g = broadcast(+, x.grad, y.grad)
    broadcast(/, gx, g, size(gx,dim))
end
