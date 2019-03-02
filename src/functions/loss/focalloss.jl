export focalloss

"""
Focal Loss for Dense Object Detection, ICCV 2017
"""
function focalloss(idx::Var, p::Var, gamma=1.0)
    ydata = focalloss(idx.data, p.data, gamma)
    y = Var(ydata, ∇focalloss!, (idx,p,gamma))
    average(y, dims=1)
end

function focalloss(idx::Vector{Int}, p::Matrix{T}, gamma) where T
    gamma = T(gamma)
    length(idx) == size(p,2) || throw("Length unmatch.")
    y = zeros(T, length(idx))
    @inbounds for i = 1:length(y)
        if idx[i] > 0
            pi = p[idx[i],i]
            y[i] = -log(pi) * (1-pi)^gamma
        end
    end
    y
end

@generated function focalloss(v::CuVector{Cint}, p::CuMatrix{T}, gamma) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void focalloss($Ct *y, int *v, $Ct *p, $Ct gamma, int dim1, int dim2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dim2) return;

        if (v[idx] > 0) {
            int i = (v[idx]-1) + dim1 * idx;
            $Ct pi = p[i];
            y[idx] = -log(pi) * pow(1-pi, gamma);
        }
        else y[idx] = 0;
    }""")
    quote
        length(v) == size(p,2) || throw("Length unmatch.")
        y = CuArray{T}(length(v))
        gdims, bdims = cudims(length(y))
        $k(gdims, bdims, pointer(y), pointer(v), pointer(p), T(gamma), size(p,1), size(p,2))
        y
    end
end

function ∇focalloss!(y::Var, idx::Var, p::Var, gamma)
    isnothing(p.grad) || ∇focalloss!(y.grad, idx.data, p.data, p.grad, gamma)
end

function ∇focalloss!(gy::Vector{T}, idx::Vector{Int}, p::Matrix{T}, gp::Matrix{T}, gamma) where T
    gamma = T(gamma)
    for i = 1:length(gy)
        pi = p[idx[i],i]
        g = -(1-pi)^gamma/pi + log(pi)*(1-pi)^(gamma-T(1))*gamma
        gp[idx[i],i] += gy[i] * g
    end
end

@generated function ∇focalloss!(gy::CuVector{T}, v::CuVector{Cint}, p::CuMatrix{T}, gp::CuMatrix{T}, gamma) where T
    Ct = cstring(T)
    k = Kernel("""
    __global__ void focalloss_grad($Ct *gy, int *v, $Ct *p, $Ct *gp, $Ct gamma, int dim1, int dim2) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= dim2) return;
        if (v[idx] <= 0) return;

        int i = (v[idx]-1) + dim1 * idx;
        $Ct pi = p[i];
        $Ct g = -pow(1-pi, gamma) / pi + log(pi) * pow(1-pi, gamma-1) * gamma;
        gp[i] += gy[idx] * g;
    }""")
    quote
        length(v) == size(p,2) == length(gy) || throw("Length unmatch.")
        gdims, bdims = cudims(length(gy))
        $k(gdims, bdims, pointer(gy), pointer(v), pointer(p), pointer(gp), T(gamma), size(p,1), size(p,2))
    end
end
