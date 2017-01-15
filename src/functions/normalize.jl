import Base.normalize

normalize(x::Var) = forward(normalize, x)

function forward{T}(::typeof(normalize), x::Matrix{T})
    z = mapreducedim(v -> v*v, +, x, 1)
    @inbounds @simd for i = 1:length(z)
        z[i] = 1 / sqrt(z[i]+1e-9)
    end
    y = x .* z
    backward!(gy, gx) = isvoid(gx) || ∇normalize!(gy, x, gx, z)
    y, backward!
end

function ∇normalize!{T}(gy::Matrix{T}, x::Matrix{T}, gx::Matrix{T}, z::Matrix{T})
    for j = 1:size(x,2)
        @inbounds @simd for i = 1:size(x,1)
            g = z[j] - x[i,j]*x[i,j] * (z[j]*z[j]*z[j])
            gx[i,j] += gy[i,j] * g
        end
    end
end

function normweight{T}(::Type{T}, row::Int, col::Int, x::Var)
    v = randn(T, row, col) * 0.05
    g = zeros(T, row)
    normalize(zerograd(v)) * zerograd(g)
end
