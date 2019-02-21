export dropout_dim, bernoulli!

function dropout_dim(x::Var, dim::Int, droprate::Float64)
    droprate == 0.0 && return x
    istraining() || return x

    dims = ntuple(i -> i == dim ? size(x,i) : 1, ndims(x))
    mask = similar(x.data, dims)
    bernoulli!(mask, droprate)
    x .* Var(mask)
end

@generated function bernoulli!(x::CuArray{T}, droprate::Float64) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void dropout_dim($Ct *x, $Ct *r, $Ct droprate, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        // x[idx] = r[idx] < droprate ? 0 : 1 / (1-droprate);
        x[idx] = r[idx] < droprate ? 0 : 1;
    }
    """)
    quote
        r = curand(T, length(x))
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, pointer(x), pointer(r), T(droprate), length(x))
        x
    end
end

#=
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
=#
