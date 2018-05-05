export lookup

function lookup(w::Var, x::Array{Int})
    configure!(w)
    y = lookup(w.data, x)
    Var(y, (lookup,w,x))
end

function lookup(w::UniMatrix{T}, x::Array{Int}) where T
    n = size(w, 1)
    y = similar(w, n*size(x,1), Base.tail(size(x))...)
    fill!(y, 0)
    for i = 1:length(x)
        x[i] <= 0 && continue
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        copy!(y, yi, w, wi, n)
    end
    y
end

function addgrad!(y::Var, ::typeof(lookup), w::Var, x::Array{Int})
    isvoid(w.grad) && return
    ∇lookup!(y.grad, w.grad, x)
end

unsafe_array(x::Array, i::Int, dims) = unsafe_wrap(Array, pointer(x,i), dims)
unsafe_array(x::CuArray, i::Int, dims) = unsafe_wrap(CuArray, pointer(x,i), dims)

function ∇lookup!(gy::UniArray{T}, gw::UniArray{T}, x::Array{Int}) where T
    n = size(gw, 1)
    for i = 1:length(x)
        x[i] <= 0 && continue
        yi = (i-1) * n + 1
        wi = (x[i]-1) * n + 1
        unsafe_gy = unsafe_array(gy, yi, (n,))
        unsafe_gw = unsafe_array(gw, wi, (n,))
        add!(unsafe_gw, unsafe_gy)
    end
end

#=
@generated function lookup(w::CuMatrix{T}, x::CuArray{Cint}) where T
    Ct = cstring(T)
    k = Kernel("""
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
        $k(gdims, bdims, pointer(y), length(y), pointer(w), pointer(x), n)
        y
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
            // gw[(x[j]-1) * n + i] += gy[idxY];
        }
    }""")
    quote
        n = size(gw, 1)
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, pointer(gy), length(gy), pointer(gw), pointer(x), n)
    end
end
=#
