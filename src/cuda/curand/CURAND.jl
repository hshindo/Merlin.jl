module CURAND

using ..CUDA

include("define.jl")

macro curand(f, args...)
    quote
        status = ccall(($f,CUDA.libcurand), Cint, $(map(esc,args)...))
        if status != 0
            throw(ERROR_MESSAGES[status])
        end
    end
end

function create_generator(rngtype::Int)
    ref = Ref{Ptr{Cvoid}}()
    @curand :curandCreateGenerator (Ptr{Ptr{Cvoid}},Cint) ref rngtype
    ref[]
end

function set_pseudo_random_generator_seed!(gen, seed::UInt64)
    @curand :curandSetPseudoRandomGeneratorSeed (Ptr{Cvoid},Culonglong) gen seed
end

function curand(::Type{T}, dims::Dims{N}) where {T,N}
    gen = getrng()
    x = CuArray{T}(dims)
    ptr = pointer(x)
    num = length(x)
    if T == UInt32
        @curand :curandGenerate (Ptr{Cvoid},Ptr{Cuint},Csize_t) gen ptr num
    elseif T == UInt64
        @curand :curandGenerateLongLong (Ptr{Cvoid},Ptr{Culonglong},Csize_t) gen ptr num
    elseif T == Float32
        @curand :curandGenerateUniform (Ptr{Cvoid},Ptr{Cfloat},Csize_t) gen ptr num
    elseif T == Float64
        @curand :curandGenerateUniformDouble (Ptr{Cvoid},Ptr{Cdouble},Csize_t) gen ptr num
    else
        throw("Unsupported type: $T.")
    end
    x
end

function curandn(::Type{T}, dims::Dims{N}, mean=0, stddev=1) where {T,N}
    gen = getrng()
    x = CuArray{T}(dims)
    ptr = pointer(x)
    n = length(x)
    if T == Float32
        @curand :curandGenerateNormal (Ptr{Cvoid},Ptr{Cfloat},Csize_t,Cfloat,Cfloat) gen ptr n mean stddev
    elseif T == Float64
        @curand :curandGenerateNormalDouble (Ptr{Cvoid},Ptr{Cdouble},Csize_t,Cdouble,Cdouble) gen ptr n mean stddev
    else
        throw("Unsupported type: $T.")
    end
    x
end

function version()
    ref = Ref{Cint}()
    @curand :curandGetVersion (Ptr{Cint},) ref
    ref[]
end

const RNGS = Dict{Int,Ptr{Cvoid}}()
# atexit(() -> @curand :curandDestroyGenerator (Ptr{Cvoid},) RNG[])

function getrng()
    dev = CUDA.getdevice()
    get!(RNGS, dev) do
        rng = create_generator(CURAND_RNG_PSEUDO_MTGP32)
        set_pseudo_random_generator_seed!(rng, rand(UInt64))
        rng
    end
end

#=
function curandlogn(rng::curandGenerator_t, T::DataType, num::Int=1; mean=T(0), stddev=T(1))
  arr = Main.CuArray(T, num)
  if T == Float32
    curandGenerateLogNormal(rng, arr, num, mean, stddev)
  elseif T == Float64
    curandGenerateLogNormalDouble(rng, arr, num, mean, stddev)
    end
  arr
end

function curandpoisson(rng::curandGenerator_t, num::Int, lambda::Float64)
  arr = Main.CuArray(UInt32, num)
  curandGeneratePoisson(rng, arr, num, lambda)
  arr
end
=#

end
