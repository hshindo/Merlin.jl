export mse

doc"""
    mse(x1, x2)

Mean Squared Error function between `x1` and `x2`.
The mean is calculated over the minibatch.
Note that the error is not scaled by 1/2.
"""
function mse(x1::Var, x2::Var; reduction=:mean)
    @assert reduction == :mean
    size(x1) == size(x2) || throw("Size unmatch.")
    ydata = mse(x1.data, x2.data)
    y = Var(ydata, ∇mse!, (x1,x2))
    average(y, dims=1, keepdims=false)
end

mse(x1::Matrix, x2::Matrix) = abs2.(x1-x2)

@generated function mse(x1::CuMatrix{T}, x2::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void mse($Ct *y, $Ct *x1, $Ct *x2, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        $Ct d = x1[idx] - x2[idx];
        y[idx] = d * d;
    }""")
    quote
        y = similar(x1)
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(x1), pointer(x2), Cint(length(y)))
        y
    end
end

∇mse!(y::Var, x1::Var, x2::Var) = ∇mse!(y.grad, x1.data, x1.grad, x2.data, x2.grad)

@generated function ∇mse!(gy::CuMatrix{T}, x1::CuMatrix{T}, gx1, x2::CuMatrix{T}, gx2) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void mse_grad($Ct *gy, $Ct *x1, $Ct *gx1, $Ct *x2, $Ct *gx2, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        $Ct g = 2 * gy[idx] * (x1[idx] - x2[idx]);
        if (gx1 != NULL) gx1[idx] += g;
        if (gx2 != NULL) gx2[idx] -= g;
    }""")
    quote
        gdims, bdims = cudims(length(gy))
        gx1 = isnothing(gx1) ? C_NULL : pointer(gx1)
        gx2 = isnothing(gx2) ? C_NULL : pointer(gx2)
        $k(gdims, bdims, pointer(gy), pointer(x1), gx1, pointer(x2), gx2, Cint(length(gy)))
    end
end

function ∇mse!(gy::Matrix{T}, x1::Matrix{T}, gx1, x2::Matrix{T}, gx2) where T
    g = gy .* (x1 - x2)
    isnothing(gx1) || axpy!(T(2), g, gx1)
    isnothing(gx2) || axpy!(T(-2), g, gx2)
end
