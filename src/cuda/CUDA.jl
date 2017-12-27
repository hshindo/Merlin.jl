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
include("interop.jl")
init_contexts()

include("NVRTC.jl")
include("array.jl")
include("arraymath.jl")
include("cublas/CUBLAS.jl")

end
