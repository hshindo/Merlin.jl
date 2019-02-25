mutable struct LayerNorm <: Functor
    g::Var
    b::Var
end

function (f::LayerNorm)(x::Var)
    m = average(x, 1)
    std = stdm(x, m.data, dims=1)
    y = layernorm(x.data, m, stdm)
    y
end

@generated function layernorm(x::CuMatrix{T}, m::CuArray{T}, std::CuArray{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void layernorm($Ct *x, $Ct *mean, $Ct *norm) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dim1*dim2) return;

        std = sqrt(norm*norm/n - avg*avg);
        y[idx] = gain / std * (x - avg);
    }
    """)
    quote
        s = sum(x, dims=1)
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, pointer(gy), pointer(x), pointer(gx), pointer(z), pointer(s),
            Cint(size(x,1)), Cint(size(x,2)))
    end
end
