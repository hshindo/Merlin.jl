import Base: max

doc"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimension.

```julia
x = Var(rand(Float32,10,5))
y = max(x, 1)
```
"""
function max(x::Var, dim::Int)
    configure!(x)
    ydata, idx = findmax(x.data, dim)
    Var(ydata, (max,x,dim,idx))
end
function max(x::Var, batchdims::Vector{Int})
    configure!(x)
    @assert sum(batchdims) == size(x,ndims(x))
    xdata = pack(x.data, batchdims, realmin(eltype(x)))
    ydata, idx = findmax(xdata, ndims(x))
    # ydata = squeeze(ydata, dim)
    Var(ydata, (max,x,batchdims,idx))
end
max(x::Node, dim::Int) = Node(max, dim)
max(x::Node, dims::Vector{Int}) = Node(max, dims)

function addgrad!(y::Var, ::typeof(max), x::Var, dim::Int, idx)
    isvoid(x.grad) && return
    ∇max!(y.grad, x.grad, dim, idx)
end

function addgrad!(y::Var, ::typeof(max), x::Var, batchdims::Vector{Int}, idx)
    isvoid(x.grad) && return
    dim = ndims(x)
    gx = pack(x.grad, batchdims, realmin(eltype(x)))
    ∇max!(y.grad, gx, dim, idx)
    add!(x.grad, unpack(gx,batchdims))
end

function ∇max!(gy::Array{T}, gx::Array{T}, dim::Int, idx::Array{Int}) where T
    @inbounds for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

function ∇max!(gy::Array{T}, gx::Array{T}, batchdims::Vector{Int}, idx::Array{Int}) where T
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

        int sub[$N];
        gy.ind2sub(sub, i);
        sub[dim] = idx[i];
        gx(sub) += gy[i];
    }
    """)
    quote
        gdims, bdims = cudims(length(idx))
        $k(gdims, bdims, gy, gx, dim-1, pointer(idx), length(idx))
    end
end
