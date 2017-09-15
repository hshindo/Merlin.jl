function size3d(x::Array, dim::Int)
    dim == 0 && return (1, length(x), 1)
    dim1, dim2, dim3 = 1, size(x,dim), 1
    for i = 1:dim-1
        dim1 *= size(x, i)
    end
    for i = dim+1:ndims(x)
        dim3 *= size(x, i)
    end
    (dim1, dim2, dim3)
end

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
function redim{T,N}(x::Array{T,N}, n::Int; pad=0)
    n == N && return x
    dims = ntuple(n) do i
        1 <= i-pad <= N ? size(x,i-pad) : 1
    end
    reshape(x, dims)
end
