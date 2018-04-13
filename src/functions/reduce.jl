doc"""
    maximum(x::Var, dim::Int)

Returns the maximum value over the given dimension.

```julia
x = Var(rand(Float32,10,5))
y = maximum(x, 1)
```
"""
function Base.maximum(x::Var, dim::Int, batchdims::Vector{Int})
    if length(batchdims) == 1 || dim != ndims(x)
        y, idx = findmax(x.data, dim)
    else
        @assert sum(batchdims) == size(x,ndims(x))
        y, idx = maximum_batch(x.data, batchdims)
    end
    Var(y, (maximum,x,dim,idx))
end
Base.maximum(x::Node, dim::Int, batchdims) = Node(maximum, x, dim, batchdims)

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

function addgrad!(y::Var, ::typeof(maximum), x::Var, dim::Int, idx)
    isvoid(x.grad) && return
    ∇maximum!(y.grad, x.grad, dim, idx)
end

function ∇maximum!(gy::Array{T}, gx::Array{T}, dim::Int, idx::Array{Int}) where T
    @inbounds for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

function reshape3d(x::CuArray, dim::Int)
    # dim == 0 && return (1, length(x), 1)
    dim1, dim2, dim3 = 1, size(x,dim), 1
    for i = 1:dim-1
        dim1 *= size(x, i)
    end
    for i = dim+1:ndims(x)
        dim3 *= size(x, i)
    end
    reshape(x, dim1, dim2, dim3)
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
