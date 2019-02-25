import LinearAlgebra: normalize, norm
export normalize

function normalize(x::Var, p::Int; dims::Int)
    @assert dims == 1 && p == 2
    z = norm(x.data, p, dims=dims)
    y = x.data ./ z
    Var(y, ∇normalize!, (x,p,dims,z))
end

function norm(x::Array{T}, p::Int; dims::Int) where T
    z = mapreduce(v -> v*v, +, x, dims=dims)
    sqrt.(z)
end

function ∇normalize!(y::Var, x::Var, p::Int, dims, z)
    isnothing(x.grad) || ∇normalize!(y.grad, x.data, x.grad, z)
end

function ∇normalize!(gy::Matrix{T}, x::Matrix{T}, gx::Matrix{T}, z::Matrix{T}) where T
    s = sum(x, dims=1)
    for j = 1:size(x,2)
        invz = T(1) / z[j]
        for i = 1:size(x,1)
            gx[i,j] += gy[i,j] * invz * (T(1) - invz * invz * s[j] * x[i,j])
        end
    end
end

@generated function ∇normalize!(gy::CuMatrix{T}, x::CuMatrix{T}, gx::CuMatrix{T}, z::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void normalize_grad($Ct *gy, $Ct *x, $Ct *gx, $Ct *z, $Ct *s, int dim1, int dim2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dim1*dim2) return;

        int j = idx / dim1;
        $Ct invz = 1 / z[j];
        gx[idx] += gy[idx] * invz * (1 - invz * invz * s[j] * x[idx]);
    }
    """)
    quote
        s = sum(x, dims=1)
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, pointer(gy), pointer(x), pointer(gx), pointer(z), pointer(s),
            Cint(size(x,1)), Cint(size(x,2)))
    end
end
