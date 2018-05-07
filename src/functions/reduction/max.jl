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
