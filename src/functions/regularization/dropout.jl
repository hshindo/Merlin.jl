export dropout, dropout_dim

doc"""
    dropout(x::Var, droprate::Float64)

Drops elements randomly with probability ``droprate`` and scales the other elements by factor ``1 / (1 - droprate)``.
"""
function dropout(x::Var, droprate::Float64)
    droprate == 0.0 && return x
    istraining() || return x
    ydata, work = dropout(x.data, droprate)
    Var(ydata, ∇dropout!, (x,droprate,work))
end
dropout(x::Node, droprate) = Node(dropout, (x,droprate))

function dropout(x::Array{T}, droprate::Float64) where T
    work = rand(T, length(x))
    scale = T(1 / (1-droprate))
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = work[i] <= droprate ? T(0) : scale*x[i]
    end
    y, work
end

dropout(x::CuArray, droprate) = CUDNN.dropout(x, droprate)

function ∇dropout!(y::Var, x::Var, droprate::Float64, work)
    isnothing(x.grad) && return
    ∇dropout!(y.grad, x.grad, droprate, work)
end

function ∇dropout!(gy::Array{T}, gx::Array{T}, droprate::Float64, work::Vector{T}) where T
    scale = T(1 / (1-droprate))
    @inbounds for i = 1:length(gx)
        gx[i] += work[i] <= droprate ? T(0) : scale*gy[i]
    end
end

∇dropout!(gy::CuArray, gx::CuArray, droprate, dropdesc) = CUDNN.∇dropout!(gy, gx, dropdesc)

function dropout_dim(x::Var, droprate::Float64)
    droprate == 0.0 && return x
    istraining() || return x
    ydata, r = dropout_dim(x.data, droprate)
    Var(ydata, ∇dropout_dim!, (x,droprate,r))
end

function ∇dropout_dim!(y::Var, x::Var, droprate::Float64, r)
    isnothing(x.grad) && return
    ∇dropout_dim!(y.grad, x.grad, droprate, r)
end

@generated function dropout_dim(x::CuArray{T}, droprate::Float64) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void dropout_dim($Ct *y, $Ct *x, float *r, float droprate, int m, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        int j = idx / m;
        float scale = 1 / (1-droprate);
        y[idx] = r[j] < droprate ? 0 : scale * x[idx];
    }
    """)
    quote
        @assert ndims(x) == 2
        y = similar(x)
        r = curand(Float32, size(x,2))
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(x), pointer(r), Float32(droprate), size(x,1), length(x))
        y, r
    end
end

@generated function ∇dropout_dim!(gy::CuArray{T}, gx::CuArray{T}, droprate, r) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void dropout_dim_grad($Ct *gy, $Ct *gx, float *r, float droprate, int m, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        int j = idx / m;
        float scale = 1 / (1-droprate);
        gx[idx] += r[j] < droprate ? 0 : scale * gy[idx];
    }
    """)
    quote
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, pointer(gy), pointer(gx), pointer(r), Float32(droprate), size(gx,1), length(gx))
    end
end
