import Base.max

"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimensions.
"""
function max(x::Var, dim::Int)
    data, idx, batchdims = findmax_batch2(x.data, dim, x.batchdims)
    Var(data, batchdims, max, (x,dim), work=idx)
end
max(x::Node, dim::Int) = Node(max, x, dim)

function findmax_batch2(x::Array{T,N}, dim::Int, batchdims::Vector{Int}) where {T,N}
    if dim == N
        front = Base.front(size(x))
        n = prod(front)
        y = T[]
        idx = Int[]

        cumdim = 0
        for i = 1:length(batchdims)
            p = pointer(x, n*cumdim+1)
            subx = unsafe_wrap(Array, p, (front...,batchdims[i]))

            val, index = findmax(subx, dim)
            @inbounds for k = 1:length(index)
                index[k] += n * cumdim
            end
            append!(y, val)
            append!(idx, index)

            cumdim += batchdims[i]
        end
        y = reshape(y, front..., length(batchdims))
        batchdims = ones(Int, length(batchdims))
        y, idx, batchdims
    else
        y, idx = findmax(x, dim)
        y, idx, batchdims
    end
end

function findmax_batch(x::Array{T,N}, dim::Int, batchdims::Vector{Int}) where {T,N}
    if dim == N
        println("debug start")
        println(size(x))
        println(dim)
        println(batchdims)
        front = Base.front(size(x))
        println(front)
        n = prod(front)
        y = Array{T}(front..., length(batchdims))
        idx = Array{Int}(size(y))
        println(size(idx))
        println(idx)

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
        ind = indmax(idx)
        if idx[ind] > length(x)
            println(ind)
            println(idx[ind])
            println(idx)
            throw("m > length(x)")
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
