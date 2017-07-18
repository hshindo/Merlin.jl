import Base.normalize

normalize(x::Var) = forward0(normalize, x)

function forward{T}(::typeof(normalize), x::Matrix{T})
    z = mapreducedim(v -> v*v, +, x, 1)
    @inbounds @simd for i = 1:length(z)
        z[i] = T(1) / sqrt(z[i]+1e-9)
    end
    y = x .* z
    backward!(gy, gx) = isvoid(gx) || ∇normalize!(gy, x, gx, z)
    y, backward!
end

function ∇normalize!{T}(gy::Matrix{T}, x::Matrix{T}, gx::Matrix{T}, z::Matrix{T})
    s = sum(x, 1)
    for j = 1:size(x,2)
        @inbounds @simd for i = 1:size(x,1)
            gx[i,j] += gy[i,j] * (z[j] - z[j]*z[j]*z[j]*s[j]*x[i,j])
        end
    end
end
