module CUDA

using Libdl
using Base.Threads
import LinearAlgebra.BLAS: scal!, axpy!, gemv, gemv!, gemm, gemm!

if Sys.iswindows()
    const libcuda = Libdl.find_library("nvcuda")
    const libcublas = Libdl.find_library(["cublas64_100","cublas64_92","cublas64_91","cublas64_90"])
    const libcurand = Libdl.find_library(["curand64_100","curand64_92","curand64_91","curand64_90"])
    const libnvml = Libdl.find_library("nvml", [joinpath(ENV["ProgramFiles"],"NVIDIA Corporation","NVSMI")])
    const libnvrtc = Libdl.find_library(["nvrtc64_100_0","nvrtc64_92","nvrtc64_91","nvrtc64_90"])
    const libcudnn = Libdl.find_library(["cudnn64_7"])
else
    const libcuda = Libdl.find_library("libcuda")
    const libcublas = Libdl.find_library("libcublas")
    const libcurand = Libdl.find_library(["libcurand"])
    const libnvml = Libdl.find_library("libnvidia-ml")
    const libnvrtc = Libdl.find_library("libnvrtc")
    const libcudnn = Libdl.find_library("libcudnn")
    const libnccl = Libdl.find_library("libnccl")
end
isempty(libcuda) && error("CUDA cannot be found.")
isempty(libcublas) && error("CUBLAS cannot be found.")
isempty(libcurand) && error("CURAND library cannot be found.")
isempty(libnvml) && error("NVML cannot be found.")
isempty(libnvrtc) && error("NVRTC cannot found.")
isempty(libcudnn) && error("CUDNN cannot be found.")
# isempty(libnccl) && @warn "NCCL cannot be found."

function versionall()
    @info CUDNN.version()
end

function checkstatus(status)
    if status != 0
        ref = Ref{Cstring}()
        ccall((:cuGetErrorString,libcuda), Cint, (Cint,Ptr{Cstring}), status, ref)
        throw(unsafe_string(ref[]))
    end
end

function version()
    ref = Ref{Cint}()
    checkstatus(ccall((:cuDriverGetVersion,libcuda), Cint, (Ptr{Cint},), ref))
    Int(ref[])
end

const API_VERSION = version()
@info "CUDA API $API_VERSION"

include("define.jl")

macro apicall(f, args...)
    f = get(DEFINE, f.value, f.value)
    quote
        status = ccall(($(QuoteNode(f)),libcuda), Cint, $(map(esc,args)...))
        checkstatus(status)
    end
end

macro unsafe_apicall(f, args...)
    f = get(DEFINE, f.value, f.value)
    quote
        ccall(($(QuoteNode(f)),libcuda), Cint, $(map(esc,args)...))
    end
end

function __init__()
    @apicall :cuInit (Cuint,) 0
    # checkstatus(ccall((:cuInit,libcuda), Cuint, (Cuint,), 0))
end

const Cptr = Ptr{Cvoid}
export cstring
cstring(::Type{Int32}) = "int"
cstring(::Type{Float32}) = "float"

const ALLOCATED = []

include("driver/device.jl")
include("driver/context.jl")
include("driver/stream.jl")
include("driver/memory.jl")
include("driver/module.jl")
include("driver/function.jl")

include("nvml/NVML.jl")
include("nvrtc/NVRTC.jl")
# using .NVML

include("pointer.jl")
include("array.jl")
include("subarray.jl")
include("kernel.jl")
include("kernels.jl")
include("arraymath.jl")
include("broadcast.jl")
include("cat.jl")
include("reduce.jl")
include("devicearray.jl")

include("allocators/atomic_malloc.jl")
include("allocators/cuda_malloc.jl")
include("allocators/mempool_malloc.jl")

include("nccl/NCCL.jl")
include("cublas/CUBLAS.jl")
include("curand/CURAND.jl")
include("cudnn/CUDNN.jl")

#using .CUBLAS, .CUDNN, .CURAND
export CUBLAS, CUDNN, CURAND

end
