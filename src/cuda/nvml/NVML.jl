module NVML

using ..CUDA
import ..CUDA: ndevices, getdevice
using Libdl

if Sys.iswindows()
    const libnvml = Libdl.find_library("nvml", [joinpath(ENV["ProgramFiles"],"NVIDIA Corporation","NVSMI")])
else
    const libnvml = Libdl.find_library("libnvidia-ml")
end
isempty(libnvml) && error("NVML cannot be found.")

const API_VERSION = Ref{String}()

function checkresult(result::Cint)
    if result != 0
        p = ccall((:nvmlErrorString,libnvml), Ptr{Cchar}, (Cint,), result)
        throw(unsafe_string(p))
    end
end

function __init__()
    result = ccall((:nvmlInit_v2,libnvml), Cint, ())
    checkresult(result)

    ref = Array{Cchar}(undef, 80)
    result = ccall((:nvmlSystemGetNVMLVersion,libnvml), Cint, (Ptr{Cchar},Cuint), ref, 80)
    checkresult(result)

    API_VERSION[] = unsafe_string(pointer(ref))
    @info "NVML $(API_VERSION[])"
end

include("define.jl")

macro nvml(f, args...)
    f = get(DEFINE, f.value, f.value)
    quote
        result = ccall(($(QuoteNode(f)),libnvml), Cint, $(map(esc,args)...))
        checkresult(result)
    end
end

macro nvml_nocheck(f, args...)
    f = get(DEFINE, f.value, f.value)
    quote
        ccall(($(QuoteNode(f)),libnvml), Cint, $(map(esc,args)...))
    end
end

include("device.jl")

end
