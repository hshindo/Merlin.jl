export max_batch

"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
function Base.max(x::Var, dim::Int)
    data, idx, batchdims = findmax_batch(x.data, dim, x.batchdims)
    Var(data, batchdims, max, (x,dim), work=idx)
end
Base.max(x::Node, dim::Int) = Node(max, x, dim)

function findmax_batch(x::Array{T,N}, dim::Int, batchdims::Vector{Int}) where {T,N}
    if dim == N
        front = Base.front(size(x))
        n = prod(front)
        y = Array{T}(front..., length(batchdims))
        idx = Array{Int}(size(y))

        cumdim = 0
        for i = 1:length(batchdims)
            p = pointer(x, n*cumdim+1)
            subx = unsafe_wrap(Array, p, (front...,batchdims[i]))
            p = pointer(y, n*(i-1)+1)
            suby = unsafe_wrap(Array, p, (front...,1))
            p = pointer(idx, n*(i-1)+1)
            subidx = unsafe_wrap(Array, p, (front...,1))

            findmax!(suby, subidx, subx)
            @inbounds for k = 1:length(subidx)
                subidx[k] += n * cumdim
            end

            cumdim += batchdims[i]
        end
        batchdims = ones(Int, length(batchdims))
        y, idx, batchdims
    else
        y, idx = findmax(x, dim)
        y, idx, batchdims
    end
end

function addgrad!(y::Var, ::typeof(max), x::Var, dim::Int)
    isvoid(x.grad) && return
    ∇max!(y.grad, x.grad, y.work)
end

function ∇max!(gy::Array{T}, gx::Array{T}, idx::Array{Int}) where T
    for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

"""
    sum(x::Var, dim::Int)

Returns the sum over the given dimension.
"""
function Base.sum(x::Var, dim::Int)
    y = Var(nothing, sum, (x,dim))
    y.data = sum(x.data, dim)
    y.df! = () -> begin
        isvoid(x.grad) || broadcast!(+, x.grad, x.grad, y.grad)
    end
    y
end
