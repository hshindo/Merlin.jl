
#=
import Base: normalize, normalize!
function normalize{T,N}(x::Array{T,N})
    z = mapreducedim(v -> v*v, +, x, N-1)
    @inbounds @simd for i = 1:length(z)
        z[i] = 1 / sqrt(z[i]+1e-9)
    end
    x .* z
end
function normalize!{T,N}(x::Array{T,N})
    z = mapreducedim(v -> v*v, +, x, N-1)
    @inbounds @simd for i = 1:length(z)
        z[i] = 1 / sqrt(z[i]+1e-9)
    end
    x .*= z
    x
end
=#

"""
    redim(x, n, [pad])
"""
function redim{T,N}(x::UniArray{T,N}, n::Int; pad=0)
    n == N && return x
    dims = ntuple(n) do i
        1 <= i-pad <= N ? size(x,i-pad) : 1
    end
    reshape(x, dims)
end
