module CUDA

if is_windows()
    const libcuda = Libdl.find_library(["nvcuda"])
else
    const libcuda = Libdl.find_library(["libcuda"])
end
isempty(libcuda) && throw("CUDA library cannot be found.")

function check_curesult(status)
    status == CUDA_SUCCESS && return nothing
    warn("CUDA error triggered from:")
    Base.show_backtrace(STDOUT, backtrace())
    ref = Ptr{UInt8}[0]
    cuGetErrorString(status, ref)
    throw(unsafe_string(ref[1]))
end

p = Cint[0]
ccall((:cuDriverGetVersion,libcuda), UInt32, (Ptr{Cint},), p)
const driver_version = Int(p[1])
const major = div(driver_version, 1000)
const minor = div(driver_version - major*1000, 10)

include("lib/$(major).$(minor)/libcuda.jl")
include("lib/$(major).$(minor)/libcuda_types.jl")

include("context.jl")
include("device.jl")
include("module.jl")
include("function.jl")
include("headers.jl")

info("CUDA driver version: $driver_version")
initctx()
infodevices()

ctype(::Type{Int64}) = :int
ctype(::Type{Float32}) = :float
ctype(::Type{Float64}) = :double

include("pointer.jl")
include("abstractarray.jl")
include("array.jl")
include("arraymath.jl")
include("reducedim.jl")
include("subarray.jl")
include("Interop.jl")

##### NVRTC #####
include("NVRTC.jl")

##### CUBLAS #####
include("CUBLAS.jl")
using .CUBLAS

##### CUDNN #####
include("cudnn/CUDNN.jl")

end
