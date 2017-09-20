import Base.max

"""
    max(x, dim::Int)

Returns the maximum value over the given dimensions.

# 👉 Example
```julia
x = Var(randn(Float32,10,5))
y = max(x, 1)
```
"""
function max(x::Var, dim::Int)
    if dim == ndims(x)
        y, idx = max_batch(x.data, x.batchdims)
        batchdims = ones(Int, length(x.batchdims))
    else
        y, idx = findmax(x.data, dim)
        batchdims = x.batchdims
    end
    Var(y, batchdims, max, (x,idx))
end

max(x::Node, dim::Int; name="max") = Node(max, x, dim, name=name)

function max_batch{T,N}(x::Array{T,N}, batchdims::Vector{Int})
    front = Base.front(size(x))
    n = prod(front)
    y = T[]
    idx = Int[]

    cumdim = 0
    for i = 1:length(batchdims)
        p = pointer(x, n*cumdim+1)
        subx = unsafe_wrap(Array, p, (front...,batchdims[i]))

        val, index = findmax(subx, N)
        @inbounds for k = 1:length(index)
            index[k] += n * cumdim
        end
        append!(y, val)
        append!(idx, index)
        cumdim += batchdims[i]
    end
    y = reshape(y, front..., length(batchdims))
    y, idx
end

function addgrad!(y::Var, ::typeof(max), x::Var, idx)
    isvoid(x.grad) || ∇max!(y.grad, x.grad, idx)
end

function ∇max!{T}(gy::Array{T}, gx::Array{T}, idx::Array{Int})
    for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end
