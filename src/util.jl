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

function Base.rand{T<:AbstractFloat,N}(low::T, high::T, dims::NTuple{N,Int})
    Array{T,N}(rand(T,dims) * (high-low) + low)
end
Base.rand{T<:AbstractFloat}(low::T, high::T, dims::Int...) = rand(low, high, dims)

Base.randn{T<:AbstractFloat,N}(::Type{T}, dims::NTuple{N,Int}) = Array{T,N}(randn(dims))
Base.randn{T<:AbstractFloat}(::Type{T}, dims::Int...) = randn(T, dims)
function Base.randn{T<:AbstractFloat,N}(low::T, high::T, dims::NTuple{N,Int})
    Array{T,N}(randn(T,dims) * (high-low) + low)
end
Base.randn{T<:AbstractFloat}(low::T, high::T, dims::Int...) = randn(low, high, dims)

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
