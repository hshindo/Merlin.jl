export max_batch

"""
    max(x, dim::Int)

Returns the maximum value over the given dimensions.

# 👉 Example
```julia
x = Var(randn(Float32,10,5))
y = max(x, 1)
```
"""
function Base.max(x::Var, dim::Int)
    y, idx = findmax(x.data, dim)
    Var(y, max, (x,idx))
end
Base.max(x::Node, dim::Int) = Node(max, x, dim)

function addgrad!(y::Var, ::typeof(max), x::Var, idx)
    isvoid(x.grad) || ∇max!(y.grad, x.grad, idx)
end

function ∇max!{T}(gy::Array{T}, gx::Array{T}, idx::Array{Int})
    for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

"""
    max_batch(x, batchsize::Vector{Int})

Batched `max` function.
The maximum values are calculated over the last dimension.

# 👉 Example
```julia
x = Var(randn(Float32,10,5))
y = max_batch(x, [2,3])
```
"""
function max_batch(x::Var, batchsize::Var)
    y, idx = max_batch(x.data, batchsize.data)
    Var(y, max_batch, (x,idx))
end
max_batch(x::Node, batchsize::Node) = Node(max_batch, x, batchsize)

function max_batch{T,N}(x::Array{T,N}, batchsize::Vector{Int})
    front = Base.front(size(x))
    n = prod(front)
    y = T[]
    idx = Int[]

    cumdim = 0
    for i = 1:length(batchsize)
        p = pointer(x, n*cumdim+1)
        subx = unsafe_wrap(Array, p, (front...,batchsize[i]))

        val, index = findmax(subx, N)
        @inbounds for k = 1:length(index)
            index[k] += n * cumdim
        end
        append!(y, val)
        append!(idx, index)
        cumdim += batchsize[i]
    end

    y = reshape(y, front..., length(batchsize))
    y, idx
end

function addgrad!(y::Var, ::typeof(max_batch), x::Var, idx)
    isvoid(x.grad) || ∇max!(y.grad, x.grad, idx)
end
