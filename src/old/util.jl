export splitdims

Base.zeros(x::Number) = zero(x)
Base.ones(x::Number) = one(x)

function pad(x::Array, N::Int)
    nd = ndims(x)
    if nd == N
        x
    elseif nd < N
        dims = fill(1,N)
        for i = 1:ndims(x)
            dims[i] = size(x,i)
        end
        reshape(x, dims...)
    else
        # TODO
        throw("Not implemented yet.")
    end
end

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

# Workaround a lack of optimization in gcc
#const exp_cst1 = 2139095040.f0
#const exp_cst2 = 0.f0

#=
@inline function exp_approx(val::Float32)
val2 = 12102203.1615614f0 * val + 1065353216.f0
val3 = val2 < exp_cst1 ? val2 : exp_cst1
val4 = val3 > exp_cst2 ? val3 : exp_cst2
val4i = floor(Int32, val4)
xu = val4i & 0x7F800000
xu2 = (val4i & 0x7FFFFF) | 0x3F800000
b = reinterpret(Float32, Int32(xu2))
xuf = reinterpret(Float32, Int32(xu))
xuf * (0.510397365625862338668154f0 + b *
(0.310670891004095530771135f0 + b *
(0.168143436463395944830000f0 + b *
(-2.88093587581985443087955f-3 + b *
1.3671023382430374383648148f-2))))
end
=#

#=
export fastexp!, normalexp!
const FASTEXP_F32 = Libdl.dlsym(library, :fastexp)
const NORMALEXP_F32 = Libdl.dlsym(library, :normalexp)
function fastexp!{T}(x::Vector{T}, y::Vector{T})
ccall(FASTEXP_F32, Void, (Ptr{T}, Ptr{T}, Cint), x, y, length(x))
end
function normalexp!{T}(x::Vector{T}, y::Vector{T})
ccall(NORMALEXP_F32, Void, (Ptr{T}, Ptr{T}, Cint), x, y, length(x))
end
=#
#=
macro fastmap(f, T, src)
quote
local src = $(esc(src))
local f = $(esc(f))
local T = $(esc(T))
dest = Array(T, length(src))
for i = 1:length(src)
dest[i] = f(src[i])
end
dest
end
end
=#
