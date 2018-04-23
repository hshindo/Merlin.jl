module CUBLAS

using ..CUDA
import ..CUDA: ndevices, getdevice

if is_windows()
    const libcublas = Libdl.find_library(["cublas64_91","cublas64_90","cublas64_80","cublas64_75"])
else
    const libcublas = Libdl.find_library("libcublas")
end
isempty(libcublas) && error("CUBLAS cannot be found.")

function init()
    ref = Ref{Ptr{Void}}()
    ccall((:cublasCreate_v2,libcublas), Cint, (Ptr{Ptr{Void}},), ref)
    h = ref[]

    ref = Ref{Cint}()
    ccall((:cublasGetVersion_v2,libcublas), Cint, (Ptr{Void},Ptr{Cint}), h, ref)
    global const API_VERSION = Int(ref[])

    info("CUBLAS API $API_VERSION")
    ccall((:cublasDestroy_v2,libcublas), Cint, (Ptr{Void},), h)
end
init()

include("define.jl")

macro cublas(f, rettypes, args...)
    f = get(DEFINE, f.args[1], f.args[1])
    quote
        status = ccall(($(QuoteNode(f)),libcublas), Cint, $(esc(rettypes)), $(map(esc,args)...))
        if status != 0
            throw(ERROR_MESSAGE[status])
        end
    end
end

function cublasop(t::Char)
    t == 'N' && return Cint(0)
    t == 'T' && return Cint(1)
    t == 'C' && return Cint(2)
    throw("Unknown cublas operation: $(t).")
end

include("handle.jl")
include("level1.jl")
include("level2.jl")
include("level3.jl")
include("extension.jl")

end
