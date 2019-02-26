export focalloss

"""
Focal Loss for Dense Object Detection, ICCV 2017
"""
function focalloss(idx::Var, p::Var, gamma)
    y = focalloss(idx.data, p.data, gamma)
    Var(y, ∇focalloss!, (idx,p,gamma))
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

function ∇focalloss!(y::Var, idx::Var, p::Var, gamma)
    isnothing(x.grad) || ∇focalloss!(y.grad, idx.data, p.data, p.grad, gamma)
end

function ∇focalloss!(gy::Vector{T}, idx::Vector{Int}, p::Matrix{T}, gp::Matrix{T}, gamma) where T
    gamma = T(gamma)
    for i = 1:length(gy)
        pi = p[idx[i],i]
        g = -(1-pi)^gamma + log(pi)*(1-pi)^(gamma-T(1))/gamma
        gx[idx[i],i] += gy[i] * g
    end
end
