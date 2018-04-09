@generated function lookup(w::CuMatrix{T}, x::CuArray{Cint}) where T
    Ct = cstring(T)
    f = CuFunction("""
    __global__ void lookup($Ct *y, int sizeY, $Ct *w, int *x, int n) {
        int idxY = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxY >= sizeY) return;

        int j = idxY / n;
        int i = idxY - n * j;
        y[idxY] = x[j] <= 0 ? 0 : w[(x[j]-1) * n + i];
    }""")
    quote
        n = size(w, 1)
        y = CuArray{T}(n*size(x,1), Base.tail(size(x))...)
        gdims, bdims = cudims(length(y))
        culaunch($f, gdims, bdims, Ptr{T}(y), length(y), Ptr{T}(w), Ptr{Cint}(x), n)
        y
    end
end

@generated function ∇lookup!(gy::CuMatrix{T}, gw::CuMatrix{T}, x::CuArray{Cint}) where T
    Ct = cstring(T)
    f = CuFunction("""
    __global__ void lookup_grad($Ct *gy, int sizeY, $Ct *gw, int *x, int n) {
        int idxY = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxY >= sizeY) return;

        int j = idxY / n;
        int i = idxY - n * j;
        if (x[j] > 0) gw[(x[j]-1) * n + i] += gy[idxY];
    }""")
    quote
        n = size(gw, 1)
        gdims, bdims = cudims(length(gy))
        culaunch($f, gdims, bdims, Ptr{T}(gy), length(gy), Ptr{T}(gw), Ptr{Cint}(x), n)
    end
end
