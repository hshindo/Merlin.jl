module CURAND

if Sys.iswindows()
    const libcurand = Libdl.find_library(["curand64_91","curand64_90","curand64_80","curand64_75"])
else
    const libcurand = Libdl.find_library(["libcurand"])
end
isempty(libcurand) && error("CURAND library cannot be found.")

include("define.jl")

macro apicall(f, args...)
    quote
        status = ccall(($f,libcurand), UInt32, $(map(esc,args)...))
        if status != CURAND_STATUS_SUCCESS
            Base.show_backtrace(STDOUT, backtrace())
            throw(errorstring(status))
        end
    end
end

function errorstring(status)
    status == CURAND_STATUS_SUCCESS && return "SUCCESS"
    status == CURAND_STATUS_VERSION_MISMATCH && return "VERSION_MISMATCH"
    status == CURAND_STATUS_NOT_INITIALIZED && return "NOT_INITIALIZED"
    status == CURAND_STATUS_ALLOCATION_FAILED && return "ALLOCATION_FAILED"
    status == CURAND_STATUS_TYPE_ERROR && return "TYPE_ERROR"
    status == CURAND_STATUS_OUT_OF_RANGE && return "OUT_OF_RANGE"
    status == CURAND_STATUS_LENGTH_NOT_MULTIPLE && return "LENGTH_NOT_MULTIPLE"
    status == CURAND_STATUS_DOUBLE_PRECISION_REQUIRED && return "DOUBLE_PRECISION_REQUIRED"
    status == CURAND_STATUS_LAUNCH_FAILURE && return "LAUNCH_FAILURE"
    status == CURAND_STATUS_PREEXISTING_FAILURE && return "PREEXISTING_FAILURE"
    status == CURAND_STATUS_INITIALIZATION_FAILED && return "INITIALIZATION_FAILED"
    status == CURAND_STATUS_ARCH_MISMATCH && return "ARCH_MISMATCH"
    status == CURAND_STATUS_INTERNAL_ERROR && return "INTERNAL_ERROR"
    throw("UNKNOWN ERROR")
end

const API_VERSION = begin
    ref = Ref{Cint}()
    @apicall :curandGetVersion (Ptr{Cint},) ref
    Int(ref[])
end
@info "CUDNN API $API_VERSION"

function _curand(rng)
    p = curandGenerator_t[0]
    curandCreateGenerator(p, rng)
    p[1]
end

###
function curng(rng_t)
  p = curandGenerator_t[0]
  curandCreateGenerator(p, rng_t)
  p[1]
end

rand_t = Float32
const _rng = curng(CURAND_RNG_PSEUDO_DEFAULT)
atexit(() -> curandDestroyGenerator(_rng))

cusrand(rng, seed) = curandSetPseudoRandomGeneratorSeed(rng, seed)
cusrand(seed) = curandSetPseudoRandomGeneratorSeed(_rng, seed)

cuoffset(rng, offset) = curandSetGeneratorOffset(rng, offset)
cuoffset(offset) = curandSetGeneratorOffset(_rng, offset)

cuorder(rng, order_t) = curandSetGeneratorOrdering(rng, order_t)
cuorder(order_t) = curandSetGeneratorOrdering(_rng, order_t)

function curng(rng_type, seed)
  rng = curng(rng_type)
  cusrand(rng, seed)
  rng
end

function curand(rng::curandGenerator_t, T::DataType, num::Int=1)
  arr = Main.CuArray(T, num)
  if  T == UInt32
    curandGenerate(rng, arr, num)
  elseif T == UInt64
    curandGenerateLongLong(rng, arr, num)
  elseif T == Float32
    curandGenerateUniform(rng, arr, num)
  elseif T == Float64
    curandGenerateUniformDouble(rng, arr, num)
  end
  arr
end

function curandn(rng::curandGenerator_t, T::DataType, num::Int=1; mean=T(0), stddev=T(1))
  arr = Main.CuArray(T, num)
  if T == Float32
    curandGenerateNormal(rng, arr, num, mean, stddev)
  elseif T == Float64
    curandGenerateNormalDouble(rng, arr, num, mean, stddev)
  end
  arr
end

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

curand(rng::curandGenerator_t, num::Int=1) = curand(rng, rand_t, num)
curand(T::DataType, num::Int=1) = curand(_rng, T, num)
curand(num::Int=1) = curand(_rng, rand_t, num)

function curandn(rng::curandGenerator_t, num::Int=1; mean=T(0), stddev=T(1))
  curandn(rng, rand_t, num, mean=mean, stdev=stdev)
end

function curandn(T::DataType, num::Int=1; mean=T(0), stddev=T(1))
  curandn(_rng, rand_t, num, mean=mean, stddev=stddev)
end

function curandn(num::Int=1; mean=0, stddev=1)
  curandn(_rng, rand_t, num, mean=rand_t(mean), stdev=rant_t(stdev))
end

function curandlogn(rng::curandGenerator_t, num::Int=1; mean=T(0), stddev=T(1))
  curandlogn(rng, rand_t, num, mean=mean, stdev=stdev)
end

function curandlogn(T::DataType, num::Int=1; mean=T(0), stddev=T(1))
  curandlogn(_rng, rand_t, num, mean=mean, stddev=stddev)
end

function curandlogn(num::Int=1; mean=0, stddev=1)
  curandlogn(_rng, rand_t, num, mean=rand_t(mean), stdev=rant_t(stdev))
end

curandpoisson(rng::curandGenerator_t, lambda::Float64) = curandpoisson(rng, 1, lambda)
curandpoisson(num::Int, lambda::Float64) = curandpoisson(_rng, num, lambda)
curandpoisson(lambda::Float64) = curandpoisson(_rng, 1, lambda)

function curandcreatedist(lambda::Float64)
  aptr = Ptr{Cvoid}[0]
  curandCreatePoissonDistribution(lambda, aptr)
  aptr[1]
end

function curandgetversion()
  aptr = Array(Int32, 1)
  curandGetVersion(aptr)
  aptr[1]
end

end
