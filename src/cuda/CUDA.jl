module CUDA

if is_windows()
    const libcuda = Libdl.find_library(["nvcuda"])
else
    const libcuda = Libdl.find_library(["libcuda"])
end
if isempty(libcuda)
    warn("CUDA library cannot be found.")
end

function init()
    ccall((:cuInit,libcuda), Cint, (Cint,), 0)
end
init()

function version()
    ref = Ref{Cint}()
    ccall((:cuDriverGetVersion,libcuda), Cint, (Ptr{Cint},), ref)
    Int(ref[])
end
const API_VERSION = version()
info("CUDA API $API_VERSION")

const mapping = Dict{Symbol,Symbol}()
if API_VERSION >= 3020
    mapping[:cuDeviceTotalMem]           = :cuDeviceTotalMem_v2
    mapping[:cuCtxCreate]                = :cuCtxCreate_v2
    mapping[:cuModuleGetGlobal]          = :cuModuleGetGlobal_v2
    mapping[:cuMemGetInfo]               = :cuMemGetInfo_v2
    mapping[:cuMemAlloc]                 = :cuMemAlloc_v2
    mapping[:cuMemAllocPitch]            = :cuMemAllocPitch_v2
    mapping[:cuMemFree]                  = :cuMemFree_v2
    mapping[:cuMemGetAddressRange]       = :cuMemGetAddressRange_v2
    mapping[:cuMemAllocHost]             = :cuMemAllocHost_v2
    mapping[:cuMemHostGetDevicePointer]  = :cuMemHostGetDevicePointer_v2
    mapping[:cuMemcpyHtoD]               = :cuMemcpyHtoD_v2
    mapping[:cuMemcpyDtoH]               = :cuMemcpyDtoH_v2
    mapping[:cuMemcpyDtoD]               = :cuMemcpyDtoD_v2
    mapping[:cuMemcpyDtoA]               = :cuMemcpyDtoA_v2
    mapping[:cuMemcpyAtoD]               = :cuMemcpyAtoD_v2
    mapping[:cuMemcpyHtoA]               = :cuMemcpyHtoA_v2
    mapping[:cuMemcpyAtoH]               = :cuMemcpyAtoH_v2
    mapping[:cuMemcpyAtoA]               = :cuMemcpyAtoA_v2
    mapping[:cuMemcpyHtoAAsync]          = :cuMemcpyHtoAAsync_v2
    mapping[:cuMemcpyAtoHAsync]          = :cuMemcpyAtoHAsync_v2
    mapping[:cuMemcpy2D]                 = :cuMemcpy2D_v2
    mapping[:cuMemcpy2DUnaligned]        = :cuMemcpy2DUnaligned_v2
    mapping[:cuMemcpy3D]                 = :cuMemcpy3D_v2
    mapping[:cuMemcpyHtoDAsync]          = :cuMemcpyHtoDAsync_v2
    mapping[:cuMemcpyDtoHAsync]          = :cuMemcpyDtoHAsync_v2
    mapping[:cuMemcpyDtoDAsync]          = :cuMemcpyDtoDAsync_v2
    mapping[:cuMemcpy2DAsync]            = :cuMemcpy2DAsync_v2
    mapping[:cuMemcpy3DAsync]            = :cuMemcpy3DAsync_v2
    mapping[:cuMemsetD8]                 = :cuMemsetD8_v2
    mapping[:cuMemsetD16]                = :cuMemsetD16_v2
    mapping[:cuMemsetD32]                = :cuMemsetD32_v2
    mapping[:cuMemsetD2D8]               = :cuMemsetD2D8_v2
    mapping[:cuMemsetD2D16]              = :cuMemsetD2D16_v2
    mapping[:cuMemsetD2D32]              = :cuMemsetD2D32_v2
    mapping[:cuArrayCreate]              = :cuArrayCreate_v2
    mapping[:cuArrayGetDescriptor]       = :cuArrayGetDescriptor_v2
    mapping[:cuArray3DCreate]            = :cuArray3DCreate_v2
    mapping[:cuArray3DGetDescriptor]     = :cuArray3DGetDescriptor_v2
    mapping[:cuTexRefSetAddress]         = :cuTexRefSetAddress_v2
    mapping[:cuTexRefGetAddress]         = :cuTexRefGetAddress_v2
    mapping[:cuGraphicsResourceGetMappedPointer] = :cuGraphicsResourceGetMappedPointer_v2
end
if API_VERSION >= 4000
    mapping[:cuCtxDestroy]               = :cuCtxDestroy_v2
    mapping[:cuCtxPopCurrent]            = :cuCtxPopCurrent_v2
    mapping[:cuCtxPushCurrent]           = :cuCtxPushCurrent_v2
    mapping[:cuStreamDestroy]            = :cuStreamDestroy_v2
    mapping[:cuEventDestroy]             = :cuEventDestroy_v2
end
if API_VERSION >= 4010
    mapping[:cuTexRefSetAddress2D]       = :cuTexRefSetAddress2D_v3
end
if API_VERSION >= 6050
    mapping[:cuLinkCreate]              = :cuLinkCreate_v2
    mapping[:cuLinkAddData]             = :cuLinkAddData_v2
    mapping[:cuLinkAddFile]             = :cuLinkAddFile_v2
end
if API_VERSION >= 6050
    mapping[:cuMemHostRegister]         = :cuMemHostRegister_v2
    mapping[:cuGraphicsResourceSetMapFlags] = :cuGraphicsResourceSetMapFlags_v2
end
if 3020 <= API_VERSION < 4010
    mapping[:cuTexRefSetAddress2D]      = :cuTexRefSetAddress2D_v2
end

macro apicall(f, args...)
    f = get(mapping, f.args[1], f.args[1])
    quote
        status = ccall(($(QuoteNode(f)),libcuda), Cint, $(map(esc,args)...))
        if status != 0
            Base.show_backtrace(STDOUT, backtrace())
            ref = Ref{Cstring}()
            ccall((:cuGetErrorString,libcuda), Cint, (Cint,Ptr{Cstring}), status, ref)
            throw(unsafe_string(ref[]))
        end
    end
end

include("device.jl")
include("context.jl")
include("stream.jl")
include("pointer.jl")
include("module.jl")
include("function.jl")
include("execution.jl")

init_contexts()

#=

include("module.jl")
include("function.jl")
include("execution.jl")

const contexts = CUcontext[]
const streams = CuStream[]

info("Initializing CUDA...")
cuInit(0)
for dev = 0:ndevices()-1
    p = CUcontext[0]
    cuCtxCreate(p, 0, dev)
    push!(contexts, p[1])
end
setdevice(0)
info("CUDA driver version: $(version())")

include("cudnn/CUDNN.jl")
include("functions/activation.jl")
=#

end
