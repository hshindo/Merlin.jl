module CUBLAS

using ..CUDA
import ..CUDA: ndevices, getdevice
import Libdl

if Sys.iswindows()
    const libcublas = Libdl.find_library(["cublas64_92","cublas64_91","cublas64_90","cublas64_80","cublas64_75"])
else
    const libcublas = Libdl.find_library("libcublas")
end
isempty(libcublas) && error("CUBLAS cannot be found.")

const API_VERSION = Ref{Int}()

function __init__()
    ref = Ref{Ptr{Cvoid}}()
    ccall((:cublasCreate_v2,libcublas), Cint, (Ptr{Ptr{Cvoid}},), ref)
    h = ref[]

    ref = Ref{Cint}()
    ccall((:cublasGetVersion_v2,libcublas), Cint, (Ptr{Cvoid},Ptr{Cint}), h, ref)
    API_VERSION[] = Int(ref[])

    @info "CUBLAS API $(API_VERSION[])"
    ccall((:cublasDestroy_v2,libcublas), Cint, (Ptr{Cvoid},), h)
end

include("define.jl")

macro cublas(f, rettypes, args...)
    f = get(DEFINE, f.value, f.value)
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
