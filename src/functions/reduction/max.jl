import Base: max

doc"""
    max(x::Var, dim::Int)

Returns the maximum value over the given dimension.

```julia
x = Var(rand(Float32,10,5))
y = max(x, 1)
```
"""
function max(x::Var, dim::Int; keepdims=true)
    ydata, idx = findmax(x.data, dims=dim)
    s = size(ydata)
    keepdims || (ydata = dropdims(ydata,dims=dim))
    Var(ydata, ∇max!, (x,dim,idx,s))
end

function max(x::Var, dims::Vector{Int})
    hdata = pack(x.data, dims, floatmin(eltype(x)))
    ydata, idx = findmax(hdata, dims=ndims(x))
    ydata = dropdims(ydata, dims=ndims(x))
    Var(ydata, ∇max!, (x,dims,idx))
end

function ∇max!(y::Var, x::Var, dim::Int, idx, s)
    isnothing(x.grad) && return
    gy = reshape(y.grad, s)
    ∇max!(gy, x.grad, dim, idx)
end

function ∇max!(y::Var, x::Var, dims::Vector{Int}, idx)
    isnothing(x.grad) && return
    gh = pack(x.grad, dims, floatmin(eltype(x)))
    gy = reshape(y.grad, Base.front(size(y))..., 1, size(y,ndims(y)))
    ∇max!(gy, gh, ndims(x), idx)
    addto!(x.grad, unpack(gh,dims))
end

function ∇max!(gy::Array{T}, gx::Array{T}, dim::Int, idx::Array) where T
    @inbounds for i = 1:length(idx)
        gx[idx[i]] += gy[i]
    end
end

@generated function ∇max!(gy::CuArray{T,N}, gx::CuArray{T,N}, dim::Int, maxidx::CuArray{Cint}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void max_grad(Array<$Ct,$N> gy, Array<$Ct,$N> gx, int dim, int *maxidx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= gy.length()) return;

        int ndidx[$N];
        gy.ndindex(ndidx, idx);
        ndidx[dim-1] = maxidx[idx] - 1;
        gx(ndidx) += gy[idx];
    }
    """)
    quote
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, gy, gx, dim, pointer(maxidx))
    end
end
