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
    y = mse(x1.data, x2.data)
    y = average(y, dims=1, keepdims=false)
    Var(y, ∇mse!, (mse,x1,x2))
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

function ∇mse!(y::Var, x1::Var, x2::Var)
    T = eltype(y)
    gx1 = isnothing(x1.grad) ? Array{T}(0,0) : x1.grad
    gx2 = isnothing(x2.grad) ? Array{T}(0,0) : x2.grad
    ∇mse!(y.grad, x1.data, gx1, x2.data, gx2)
end

function ∇mse!(gy::Vector{T}, x1::CuMatrix{T}, gx1::CuMatrix{T}, x2::CuMatrix{T}, gx2::CuMatrix{T}) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void mse_grad($Ct *gy, $Ct *x1, $Ct *gx1, $Ct *x2, $Ct *gx2, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;

        $Ct g = 2 * gy[idx] * (x1[idx]-x2[idx]);
        if () gx1[idx] += g;
        if () gx2[idx] -= g;
    }""")
    quote
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, pointer(gy), pointer(x1), pointer(gx1), pointer(x2), pointer(gx2), Cint(length(gy)))
    end
end

function ∇mse!(gy::Vector{T}, x1::Matrix{T}, gx1::Matrix{T}, x2::Matrix{T}, gx2::Matrix{T}) where T
    for j = 1:size(x1,2)
        for i = 1:size(x1,1)
            g = gy[j] * (x1[i,j]-x2[i,j]) * 2 / size(x1,1)
            isempty(gx1) || (gx1[i,j] += g)
            isempty(gx2) || (gx2[i,j] -= g)
        end
    end
end
