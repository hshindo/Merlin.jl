export argmax, @fastmap

function argmax(x, dim::Int)
  _, index = findmax(x, dim)
  ind2sub(size(x), vec(index))[dim]
end

function Base.rand{T,N}(::Type{T}, low::Float64, high::Float64, dims::NTuple{N,Int})
  # sqrt(6 / (dims[1]+dims[2]))
  a = rand(T, dims) * (high-low) + low
  convert(Array{T,N}, a)
end

Base.randn{T}(::Type{T}, dims...) = convert(Array{T}, randn(dims))

empty{T}(::Type{Array{T,1}}) = Array(T, 0)
empty{T}(::Type{Array{T,2}}) = Array(T, 0, 0)
empty{T}(::Type{Array{T,3}}) = Array(T, 0, 0, 0)
empty{T}(::Type{Array{T,4}}) = Array(T, 0, 0, 0, 0)
empty{T}(::Type{Array{T,5}}) = Array(T, 0, 0, 0, 0, 0)
empty{T}(::Type{Array{T,6}}) = Array(T, 0, 0, 0, 0, 0, 0)

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
