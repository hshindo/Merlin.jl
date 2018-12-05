export average
import Statistics: mean

#=
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
=#

doc"""
    average(x, dim::Int)

Computes the average over the given dimension.
"""
function average(x::Var, dim::Int)
    ydata = mean(x.data, dims=dim)
    Var(ydata, ∇average!, (x,dim))
end

function average(x::Var, dims::Vector{Int})
    hdata = pack(x.data, dims, 0)
    ydata = sum(hdata, dims=ndims(x))
    ydata = dropdims(ydata, dims=ndims(x))
    coef = [eltype(x)(1) / dims[i] for i=1:length(dims)]
    coef = reshape(coef, ntuple(_ -> 1, ndims(x)-1)..., length(coef))
    ydata = ydata .* coef
    Var(ydata, ∇average!, (x,dims))
end

function ∇average!(y::Var, x::Var, dim::Int)
    isnothing(x.grad) && return
    broadcast_addto!(1, x.grad, 1/size(x,dim), y.grad)
end

@generated function mean_a(gy::CuArray{T,N}, gx::CuArray{T,N}, dims::Vector{Int}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void mean_grad(Array<$Ct,$N> gy, Array<$Ct,$N> gx, int dim, int *maxidx) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= gy.length()) return;

        int ndidx[$N];
        gy.ndindex(ndidx, idx);
        ndidx[dim-1] = maxidx[idx] - 1;
        gx(ndidx) += gy[idx];
    }
    """)
    quote
        d_dims = CuArray(dims)
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, gy, gx, dim, pointer(maxidx))
    end
end
