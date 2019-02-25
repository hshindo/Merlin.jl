export stdm
import Statistics.stdm

function stdm(x::Var, m; dims::Int)
    y = stdm(x.data, m, dims=dims)
    Var(y, ∇stdm!, (x,m))
end

function stdm(x::Matrix{T}, m::Array{T}; dims::Int) where T
    x = x .- m
    y = sum(x.*x, dims=dims) / size(x,1)
    sqrt.(y .+ T(1e-9))
end

@generated function stdm(x::CuMatrix{T}, m::CuArray{T}; dims::Int) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void stdm($Ct *y, $Ct *m, $Ct *norm, int dim1, int dim2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dim2) return;
        y[idx] = sqrt(norm[idx]*norm[idx]/dim1 - m[idx]*m[idx] + 1e-9);
    }
    """)
    quote
        l2norm = norm(x, 2, dims=1)
        y = similar(m)
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(m), pointer(l2norm), Cint(size(x,1)), Cint(size(x,2)))
        y
    end
end

function ∇stdm!(y::Var, x::Var, m)
    isnothing(x.grad) || ∇stdm!(y.data,y.grad,x.data,x.grad,m)
end

function ∇stdm!(y::Array, gy::Array, x::Array, gx::Array, m::Array)
    g = (x .- m) ./ y / size(x,1)
    addto!(gx, gy .* g)
end

@generated function ∇stdm!(y::CuArray{T}, gy::CuArray{T}, x::CuArray{T}, gx::CuArray{T}, m::CuArray{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void stdm_grad($Ct *y, $Ct *gy, $Ct *x, $Ct *gx, $Ct *m, int dim1, int dim2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dim1*dim2) return;

        int j = idx / dim1;
        int i = idx - j * dim1;
        gx[idx] += gy[j] * (x[idx] - m[j]) / (y[j] * dim1);
    }
    """)
    quote
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, pointer(y), pointer(gy), pointer(x), pointer(gx),
            pointer(m), Cint(size(x,1)), Cint(size(x,2)))
    end
end
