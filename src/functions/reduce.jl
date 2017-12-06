import Base: max, mean, sum
export max_batch

doc"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimension.

# 👉 Example
```julia
x = Var(rand(Float32,10,5))
y = max(x, 1)
```
"""
function max(x::Var, dim::Int)
    y, idx = findmax(x.data, dim)
    Var(y, max, (x,idx))
end
max(x::Node, dim::Int; name="") = Node(max, (x,dim), name)

function addgrad!(y::Var, ::typeof(max), x::Var, idx)
    isvoid(x.grad) || ∇max!(y.grad, x.grad, idx)
end

function ∇max!(gy::Array{T}, gx::Array{T}, idx::Array{Int}) where T
    for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

doc"""
    max_batch(x::Var, dims::Vector{Int})
"""
function max_batch(x::Var, dims::Vector{Int})
    y, idx = max_batch(x.data, dims)
    Var(y, max, (x,idx))
end
max_batch(x::Node, dims::Vector{Int}; name="") = Node(max_batch, (x,dims), name)

function max_batch(x::Array{T,N}, dims::Vector{Int}) where {T,N}
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

function addgrad!(y::Var, ::typeof(max_batch), x::Var, idx)
    isvoid(x.grad) || ∇max!(y.grad, x.grad, idx)
end

doc"""
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

doc"""
    sum(x::Var, dim::Int)

Returns the sum over the given dimension.
"""
function sum(x::Var, dim::Int)
    throw("Not implemented yet.")
end
