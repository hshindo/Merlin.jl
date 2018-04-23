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
    configure!(x)
    y, idx = findmax(x.data, dim)
    Var(y, (max,x,dim,idx))
end

function addgrad!(y::Var, ::typeof(max), x::Var, dim::Int, idx)
    isvoid(x.grad) && return
    ∇max!(y.grad, x.grad, dim, idx)
end

function ∇max!(gy::Array{T}, gx::Array{T}, dim::Int, idx::Array{Int}) where T
    @inbounds for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

@generated function ∇max!(gy::CuArray{T,N}, gx::CuArray{T,N}, dim::Int, idx::CuArray{Cint}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void max_grad(Array<$Ct,$N> gy, Array<$Ct,$N> gx, int dim, int *idx, int length) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= length) return;

        int ndIdx[$N];
        gy.idx2ndIdx(ndIdx, i);
        ndIdx[dim] = idx[i];
        gx(ndIdx) += gy[i];
    }
    """)
    quote
        gdims, bdims = cudims(length(idx))
        $k(gdims, bdims, gy, gx, dim-1, pointer(idx), length(idx))
    end
end

doc"""
    max_batch(x::Var, batchdims)

Returns the maximum value over the batch dimension.

```julia
x = Var(rand(Float32,10,5))
y = max_batch(x, (3,2))
```
"""
function max_batch(x::Var, batchdims)
    N = ndims(x)
    @assert sum(batchdims) == size(x,N)

    xsize = Int[size(x)...]
    s = stride(x, N)
    cumdim = 0
    y = similar(x.data)
    for i = 1:length(batchdims)
        d = batchdims[i]
        xsize[N] = d
        xx = unsafe_array(x.data)
        yy, idx = findmax(xx)

        cumdim += d
    end


    xs = unsafe_split(x.data, batchdims)
    for x in xs
        max(x, ndims(x))
    end
    y, idx = max_batch(x.data, batchdims)
end

function unsafe_split2(x::UniArray{T,N}, dim::Int, dims) where {T,N}
    sum(dims) == size(x,N) || throw("Invalid splitdims: $dims.")
    length(dims) == 1 && return [x]
    xsize = [size(x)...]
    m = length(x) ÷ size(x,N)
    cumdim = 0
    ys = typeof(x)[]
    for d in dims
        xsize[N] = d
        unsafe_array(x, m*cumdim+1, (front...,d))

        p = pointer(x, m*cumdim+1)
        y = unsafe_wrap(Array, p, (front...,d))
        push!(ys, y)
        cumdim += d
    end
    ys
end
