export ptanh

"""
    Penalized tanh.
"""
ptanh(x::Var, a=0.25) = Var(ptanh(x.data,eltype(x)(a)), ∇ptanh!, (x,a))
ptanh(x::Array{T}, a::T) where T = ptanh.(x,a)
ptanh(x::T, a::T) where T = x > T(0) ? tanh(x) : a*tanh(x)

@generated function ptanh(x::CuArray{T}, a::T) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void ptanh($Ct *y, $Ct *x, $Ct a, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        y[idx] = x[idx] > 0 ? tanh(x[idx]) : a*tanh(x[idx]);
    }
    """)
    quote
        y = similar(x)
        gdims, bdims = cudims(length(x))
        $k(gdims, bdims, pointer(y), pointer(x), a, length(y))
        y
    end
end

function ∇ptanh!(y::Var, x::Var, a)
    isnothing(x.grad) && return
    ∇ptanh!(y.data, y.grad, x.data, x.grad, eltype(x)(a))
end

function ∇ptanh!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}, a::T) where T
    @inbounds for i = 1:length(gx)
        if x[i] > T(0)
            g = T(1) - y[i] * y[i]
        else
            yy = y[i] / a
            g = a * (T(1) - yy * yy)
        end
        gx[i] += gy[i] * g
    end
end

@generated function ∇ptanh!(y::CuArray{T}, gy::CuArray{T}, x::CuArray{T}, gx::CuArray{T}, a::T) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void ptanh_grad($Ct *y, $Ct *gy, $Ct *x, $Ct *gx, $Ct a, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        $Ct g = 0.0;
        if (x[idx] > 0) g = 1 - y[idx] * y[idx];
        else {
            $Ct yy = y[idx] / a;
            g = a * (1 - yy * yy);
        }
        gx[idx] += gy[idx] * g;
    }
    """)
    quote
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(gy), pointer(x), pointer(gx), a, length(y))
    end
end
