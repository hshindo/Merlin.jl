export max_batch

doc"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimension.

```julia
x = Var(rand(Float32,10,5))
y = max(x, 1)
```
"""
function Base.max(x::Var, dim::Int)
    y, idx = findmax(x.data, dim)
    Var(y, (max,x,dim,idx))
end

function max_batch(x::Var, batchdims::Vector{Int})
    @assert sum(batchdims) == size(x,ndims(x))
    xs = unsafe_split(x.data, batchdims)
    for x in xs
        max(x, ndims(x))
    end
    y, idx = max_batch(x.data, batchdims)
end

function Base.max(x::Var, dim::Int, batchdims::Vector{Int})
    if length(batchdims) == 1 || dim != ndims(x)
        y, idx = findmax(x.data, dim)
    else
        @assert sum(batchdims) == size(x,ndims(x))
        y, idx = maximum_batch(x.data, batchdims)
    end
    Var(y, (maximum,x,dim,idx))
end
max_batch(x::Node, dim::Int, batchdims) = Node(max_batch, x, dim, batchdims)

function max_batch(x::UniArray{T,N}, dims::Vector{Int}) where {T,N}
    xs = unsafe_split(x, dims)
    for x in xs
        findmax(x.data, dim)
    end
end

function max_batch2(x::Array{T,N}, dims::Vector{Int}) where {T,N}
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

function ∇max!(gy::Array{T}, gx::Array{T}, dim::Int, idx::Array{Int}) where T
    @inbounds for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
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
