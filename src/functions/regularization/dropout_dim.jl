export dropout_dim, bernoulli!

function dropout_dim(x::Var, dim::Int, droprate::Float64, scaling::Bool)
    droprate == 0.0 && return x
    istraining() || return x

    dims = ntuple(i -> i == dim ? size(x,i) : 1, ndims(x))
    mask = similar(x.data, dims)
    bernoulli!(mask, droprate, scaling)
    x .* Var(mask)
end

@generated function bernoulli!(x::CuArray{T}, droprate::Float64, scaling::Bool) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void dropout_dim($Ct *x, $Ct *r, $Ct droprate, int scaling, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        if (scaling == 1) x[idx] = r[idx] < droprate ? 0 : 1 / (1-droprate);
        else x[idx] = r[idx] < droprate ? 0 : 1;
    }
    """)
    quote
        r = curand(T, length(x))
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, pointer(x), pointer(r), T(droprate), Cint(scaling), length(x))
        x
    end
end

export replace1d
function replace1d(x::Var, p::Float64, v::Var)
    p == 0.0 && return x
    istraining() || return x
    y, r = replace1d(x.data, p, v.data)
    Var(y, ∇replace1d!, (x,p,v,r))
end

@generated function replace1d(x::CuMatrix{T}, p::Float64, v::CuVector{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void replace1d($Ct *y, $Ct *x, $Ct *r, $Ct p, $Ct *v, int dim1, int dim2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dim1*dim2) return;

        int j = idx / dim1;
        int i = idx - j * dim1;
        y[idx] = r[j] < p ? v[i] : x[idx];
    }
    """)
    quote
        @assert length(v) == size(x,1)
        y = similar(x)
        r = curand(T, size(x,2))
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, pointer(y), pointer(x), pointer(r), T(p), pointer(v),
            Cint(size(x,1)), Cint(size(x,2)))
        y, r
    end
end

function ∇replace1d!(y::Var, x::Var, p, v::Var, r)
    isnothing(x.grad) || ∇replace1d!(y.grad,x.grad,p,v.grad,r)
end

@generated function ∇replace1d!(gy::CuMatrix{T}, gx::CuMatrix{T}, p::Float64, gv::CuVector{T}, r::CuVector{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void replace1d_grad($Ct *gy, $Ct *gx, $Ct *r, $Ct p, $Ct *gv, int dim1, int dim2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dim1*dim2) return;

        int j = idx / dim1;
        int i = idx - j * dim1;
        if (r[j] < p) atomicAdd(&gv[i], gy[idx]);
        else gx[idx] += gy[idx];
    }
    """)
    quote
        @assert length(gv) == size(gx,1)
        gdims, bdims = cudims(length(gx))
        $k(gdims, bdims, pointer(gy), pointer(gx), pointer(r), T(p), pointer(gv),
            Cint(size(gx,1)), Cint(size(gx,2)))
    end
end
