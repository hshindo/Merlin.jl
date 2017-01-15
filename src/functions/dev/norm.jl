import Base.vecnorm

function vecnorm(x::Var, p=1)
    y = vecnorm(x.data, p)
    df(gy) = isconst(x) || ∇vecnorm!(x, p, x.grad, gy)
    Var(y, [x], df)
end

function vecnorm{T}(x::Array{T}, p)
    y = T(0)
    @inbounds @simd for i = 1:length(x)
        y += x[i]^p
    end
    y
end

function ∇vecnorm!{T,P}(x::Array{T}, p::P, gx::Array{T}, gy::T)
    @inbounds @simd for i = 1:length(x)
        gx[i] += gy * (p*x[i])^(p-1)
    end
    y
end
