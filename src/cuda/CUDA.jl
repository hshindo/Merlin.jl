module CUDA

if is_windows()
    const libcuda = Libdl.find_library(["nvcuda"])
else
    const libcuda = Libdl.find_library(["libcuda"])
end
if isempty(libcuda)
    warn("CUDA library cannot be found.")
end

ccall((:cuInit,libcuda), Cint, (Cint,), 0)

const API_VERSION = begin
    ref = Ref{Cint}()
    ccall((:cuDriverGetVersion,libcuda), Cint, (Ptr{Cint},), ref)
    Int(ref[])
end
info("CUDA API $API_VERSION")

include("define.jl")

macro apicall(f, args...)
    f = get(define, f.args[1], f.args[1])
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

include("NVRTC.jl")

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
