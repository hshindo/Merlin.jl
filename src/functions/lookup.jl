export lookup

function lookup(w::Var, x::Var)
    ydata = lookup(w.data, x.data)
    Var(ydata, ∇lookup!, (w,x))
end
lookup(w::Node, x) = Node(lookup, (w,x))
lookup(w, x::Node) = Node(lookup, (w,x))

function lookup(w::Matrix{T}, x::Array{Int}) where T
    s = Base.setindex(size(x), size(x,1)*size(w,1), 1)
    y = fill!(similar(w,s...), 0)
    n = size(w, 1)
    for i = 1:length(x)
        x[i] == 0 && continue
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        copyto!(y, yi, w, wi, n)
    end
    y
end

@generated function lookup(w::CuMatrix{T}, x::CuArray{Cint}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void lookup($Ct *y, int sizeY, $Ct *w, int *x, int n) {
        int idxY = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxY >= sizeY) return;

        int j = idxY / n;
        int i = idxY - n * j;
        y[idxY] = x[j] == 0 ? 0 : w[(x[j]-1) * n + i];
    }""")
    quote
        n = size(w, 1)
        ysize = Base.setindex(size(x), n*size(x,1), 1)
        y = similar(w, ysize)
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), length(y), pointer(w), pointer(x), n)
        y
    end
end

function ∇lookup!(y::Var, w::Var, x::Var)
    isnothing(w.grad) && return
    ∇lookup!(y.grad, w.grad, x.data)
end

function ∇lookup!(gy::Array, gw::Array, x::Array{Int})
    n = size(gw, 1)
    for i = 1:length(x)
        x[i] == 0 && continue
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        addto!(gw, wi, gy, yi, n)
    end
end

@generated function ∇lookup!(gy::CuArray{T}, gw::CuArray{T}, x::CuArray{Cint}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void lookup_grad($Ct *gy, int sizeY, $Ct *gw, int *x, int n) {
        int idxY = blockIdx.x * blockDim.x + threadIdx.x;
        if (idxY >= sizeY) return;

        int j = idxY / n;
        int i = idxY - n * j;
        if (x[j] > 0) {
            int idxW = (x[j]-1) * n + i;
            atomicAdd(&gw[idxW], gy[idxY]);
        }
    }""")
    quote
        n = size(gw, 1)
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, pointer(gy), length(gy), pointer(gw), pointer(x), n)
    end
end
