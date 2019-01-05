import LinearAlgebra.normalize

function normalize(x::Var, dim::Int)
    Var(normalize(x.data), ∇normalize!, (x,))
end

function normalize(x::Matrix{T})
    z = mapreducedim(v -> v*v, +, x, 1)
    x ./ z
end

function ∇normalize!(y::Var, x::Var)
    isnothing(x.grad) && return
    ∇normalize!(y.grad, x.grad)
end

function ∇normalize!(gy::Matrix{T}, x::Matrix{T}, gx::Matrix{T}, z::Matrix{T}) where T
    s = sum(x, 1)
    for j = 1:size(x,2)
        @inbounds @simd for i = 1:size(x,1)
            gx[i,j] += gy[i,j] * (z[j] - z[j]*z[j]*z[j]*s[j]*x[i,j])
        end
    end
end
