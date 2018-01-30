export lookup

function lookup(w::Var, x::UniVector)
    y = lookup(w.data, x)
    Var(y, (lookup,w,x))
end

function lookup(w::Matrix{T}, x::Vector{Int}) where T
    n = size(w, 1)
    y = similar(w, n, length(x))
    for i = 1:length(x)
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        copy!(y, yi, w, wi, n)
    end
    y
end

@generated function lookup(w::CuMatrix{T}, x::CuVector{Cint}) where T
    Ct = cstring(T)
    f = CuFunction("""
    __global__ void lookup($Ct *y, int sizeY, $Ct *w, int *x, int n) {
        int idxY = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxY >= sizeY) return;

        int j = idxY / n;
        int i = idxY - n * j;
        y[idxY] = w[(x[j]-1) * n + i];
    }""")
    quote
        n = size(w, 1)
        y = CuArray{T}(n, length(x))
        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, y.ptr, length(y), w.ptr, x.ptr, n)
        y
    end
end

function addgrad!(y::Var, ::typeof(lookup), w::Var, x)
    isvoid(w.grad) && return
    ∇lookup!(y.grad, w.grad, x)
end

function ∇lookup!(gy::Array{T}, gw::Array{T}, x::Vector{Int}) where T
    n = size(gw, 1)
    for i = 1:length(x)
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        py = pointer(gy, yi)
        pw = pointer(gw, wi)
        BLAS.axpy!(n, T(1), py, 1, pw, 1)
    end
end

@generated function ∇lookup!(gy::CuMatrix{T}, gw::CuMatrix{T}, x::CuVector{Cint}) where T
    Ct = cstring(T)
    f = CuFunction("""
    __global__ void lookup_grad($Ct *gy, int sizeY, $Ct *gw, int *x, int n) {
        int idxY = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxY >= sizeY) return;

        int j = idxY / n;
        int i = idxY - n * j;
        gw[(x[j]-1) * n + i] += gy[idxY];
    }""")
    quote
        n = size(gw, 1)
        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, gy.ptr, length(gy), gw.ptr, x.ptr, n)
    end
end
