export max_batch

doc"""
    maximum(x::Var, dim::Int)

Returns the maximum value over the given dimension.

```julia
x = Var(rand(Float32,10,5))
y = maximum(x, 1)
```
"""
function Base.max(x::Var, dim::Int)
    y, idx = findmax(x.data, dim)
    Var(y, (max,x,dim,idx))
end
Base.max(x::Node, dim::Int; name="") = Node(max, (x,dim), name)

function addgrad!(y::Var, ::typeof(max), x::Var, dim, idx)
    isvoid(x.grad) && return
    ∇max!(y.grad, x.grad, dim, idx)
end

function ∇max!(gy::Array{T}, gx::Array{T}, dim::Int, idx::Array{Int}) where T
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

@generated function ∇max!(gy::CuArray{T}, gx::CuArray{T}, dim::Int, idx::CuArray{Cint}) where T
    Ct = cstring(T)
    f = CuFunction("""
    $(LibCUDA.Array_h)
    __global__ void f(Array<$Ct,3> gy, Array<$Ct,3> gx, int *idx, int length) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= length) return;

        int ndIdx[3];
        gy.idx2ndIdx(ndIdx, i);
        ndIdx[1] = idx[i];
        gx(ndIdx) += gy[i];
    }
    """)
    quote
        gy3d = reshape3d(gy, dim)
        gx3d = reshape3d(gx, dim)
        gdims, bdims = cudims(length(idx))
        culaunch($f, gdims, bdims, gy3d, gx3d, idx.ptr, length(idx))
    end
end

doc"""
    max_batch(x::Var, dims::Vector{Int})
"""
function max_batch(x::Var, dims::Vector{Int})
    @assert sum(dims) == size(x)[end]
    y, idx = max_batch(x.data, dims)
    Var(data, (max_batch,x,dims,idx))
end
max_batch(x::Node, dims::Node; name="") = Node(maximum_batch, (x,dims), name)

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

function addgrad!(y::Var, ::typeof(max_batch), x::Var, dim, idx)
    isvoid(x.grad) && return
    ∇max!(y.grad, x.grad, dim, idx)
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
