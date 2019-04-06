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
function average(x::Var; dims::Int, keepdims=true)
    ydata = mean(x.data, dims=dims)
    s = size(ydata)
    keepdims || (ydata = dropdims(ydata,dims=dims))
    Var(ydata, ∇average!, (x,dims,s))
end

function average(x::Var, dims::Vector{Int})
    h = pack(x, dims, 0)
    h = sum(h, ndims(x), keepdims=false)
    coef = map(d -> eltype(x)(1)/d, dims)
    coef = reshape(coef, ntuple(_ -> 1, ndims(x)-1)..., length(coef))
    coef = todevice!(Var(coef))
    h .* coef
end

function average(x::Var, dims::Vector{Int}, d_dims::Var)
    h = pack(x, dims, 0)
    h = sum(h, ndims(x), keepdims=false)
    h .* d_dims
end

function ∇average!(y::Var, x::Var, dims::Int, s)
    isnothing(x.grad) && return
    gy = reshape(y.grad, s)
    broadcast_addto!(1, x.grad, 1/size(x,dims), gy)
end

#=
function ∇average!(y::Var, x::Var, dims::Vector{Int})
    isnothing(x.grad) && return
    throw("Not implemented yet.")
    broadcast_addto!(1, x.grad, 1/size(x,dim), y.grad)
end

@generated function ∇average!(gy::CuArray{T,N}, gx::CuArray{T,N}, dims::Vector{Int}) where {T,N}
    Ct = cstring(T)
    k = Kernel("""
    __global__ void average_grad(Array<$Ct,$N> gy, Array<$Ct,$N> gx, int dim, int *maxidx) {
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
=#
