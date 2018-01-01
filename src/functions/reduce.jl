export maximum_batch

doc"""
    maximum(x::Var, dim::Int)

Returns the maximum value over the given dimension.

```julia
x = Var(rand(Float32,10,5))
y = maximum(x, 1)
```
"""
function Base.maximum(x::Var, dim::Int)
    data, idx = findmax(x.data, dim)
    y = Var(data, (max,x,idx))
    y.∇! = () -> begin
        isvoid(x.grad) || ∇maximum!(y.grad, x.grad, idx)
    end
    y
end
Base.maximum(x::Node, dim::Int; name="") = Node(maximum, (x,dim), name)

function ∇maximum!(gy::Array{T}, gx::Array{T}, idx::Vector{Int}) where T
    @inbounds for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

doc"""
    maximum_batch(x::Var, dims::Vector{Int})
"""
function maximum_batch(x::Var, dims::Vector{Int})
    @assert sum(dims) == size(x)[end]
    data, idx = maximum_batch(x.data, dims)
    y = Var(data, (max_batch,x,idx))
    y.∇! = () -> begin
        isvoid(x.grad) || ∇maximum!(y.grad, x.grad, idx)
    end
end
maximum_batch(x::Node, dims::Node; name="") = Node(maximum_batch, (x,dims), name)

function maximum_batch(x::Array{T,N}, dims::Vector{Int}) where {T,N}
    front = Base.front(size(x))
    n = prod(front)
    y = T[]
    idx = Int[]

    cumdim = 0
    for i = 1:length(dims)
        p = pointer(x, n*cumdim+1)
        subx = unsafe_wrap(Array, p, (front...,dims[i]))

        val, index = findmax(subx, N)
        for k = 1:length(index)
            index[k] += n * cumdim
        end
        append!(y, val)
        append!(idx, index)
        cumdim += dims[i]
    end
    y = reshape(y, front..., length(dims))
    y, idx
end

doc"""
    mean(x, dim::Int)

Computes the average over the given dimension.
"""
function Base.mean(x::Var, dim::Int)
    if dim == ndims(x)
        y = mean_batch(x.data, x.batchdims)
        batchdims = ones(Int, length(x.batchdims))
    else
        y = mean(x.data, dim)
        batchdims = x.batchdims
    end
    Var(y, batchdims, mean, (x,dim))
end

Base.mean(x::Node, dim::Int; name="") = Node(mean, (x,dim), name)

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

doc"""
    sum(x::Var, dim::Int)

Returns the sum over the given dimension.
"""
function Base.sum(x::Var, dim::Int)
    throw("Not implemented yet.")
end
