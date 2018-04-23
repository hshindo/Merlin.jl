module NVML

using ..CUDA
import ..CUDA: ndevices, getdevice

if is_windows()
    const libnvml = Libdl.find_library("nvml", [joinpath(ENV["ProgramFiles"],"NVIDIA Corporation","NVSMI")])
else
    const libnvml = Libdl.find_library("libnvidia-ml")
end
isempty(libnvml) && error("NVML cannot be found.")

function checkresult(result::Cint)
    if result != 0
        p = ccall((:nvmlErrorString,libnvml), Ptr{Cchar}, (Cint,), result)
        throw(unsafe_string(p))
    end
end

function init()
    result = ccall((:nvmlInit_v2,libnvml), Cint, ())
    checkresult(result)

    ref = Array{Cchar}(80)
    result = ccall((:nvmlSystemGetNVMLVersion,libnvml), Cint, (Ptr{Cchar},Cuint), ref, 80)
    checkresult(result)

    const API_VERSION = unsafe_string(pointer(ref))
    info("NVML $API_VERSION")
end
init()

include("define.jl")

macro nvml(f, args...)
    f = get(DEFINE, f.args[1], f.args[1])
    quote
        result = ccall(($(QuoteNode(f)),libnvml), Cint, $(map(esc,args)...))
        checkresult(result)
    end
end

macro nvml_nocheck(f, args...)
    f = get(DEFINE, f.args[1], f.args[1])
    quote
        ccall(($(QuoteNode(f)),libnvml), Cint, $(map(esc,args)...))
    end
end

include("device.jl")

end
