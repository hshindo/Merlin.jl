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

1.3671023382430374383648148f-2

# Workaround a lack of optimization in gcc
const exp_cst1 = 2139095040.f0
const exp_cst2 = 0.f0

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
