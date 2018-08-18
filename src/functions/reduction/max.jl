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
    ydata, idx = findmax(x.data, dims=dim)
    Var(ydata, (max,x,dim,idx))
end
function max(x::Var, batchdims::Vector{Int})
    @assert sum(batchdims) == size(x,ndims(x))
    configure!(x)

    xdata = pack(x.data, batchdims, floatmin(eltype(x)))
    ydata, idx = findmax(xdata, dims=ndims(x))
    # ydata = dropdims(ydata, dims=dim)
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
    gx = pack(x.grad, batchdims, floatmin(eltype(x)))
    ∇max!(y.grad, gx, ndims(x), idx)
    addto!(x.grad, unpack(gx,batchdims)) # TODO: more memory-efficient implementation
end

function ∇max!(gy::Array{T}, gx::Array{T}, dim::Int, idx::Array) where T
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
        $k(gdims, bdims, gy, gx, dim-1, rawpointer(idx), length(idx))
    end
end
