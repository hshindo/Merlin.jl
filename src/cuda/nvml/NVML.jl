module NVML

using ..CUDA

function checkresult(result::Cint)
    if result != 0
        p = ccall((:nvmlErrorString,libnvml), Ptr{Cchar}, (Cint,), result)
        throw(unsafe_string(p))
    end
end

include("define.jl")

macro nvml(f, args...)
    f = get(DEFINE, f.value, f.value)
    quote
        result = ccall(($(QuoteNode(f)),CUDA.libnvml), Cint, $(map(esc,args)...))
        checkresult(result)
    end
end

macro nvml_nocheck(f, args...)
    f = get(DEFINE, f.value, f.value)
    quote
        ccall(($(QuoteNode(f)),CUDA.libnvml), Cint, $(map(esc,args)...))
    end
end

function __init__()
    @nvml :nvmlInit_v2 ()
end

function version()
    ref = Array{Cchar}(undef, 80)
    @nvml :nvmlSystemGetNVMLVersion (Ptr{Cchar},Cuint) ref 80
    unsafe_string(pointer(ref))
end

include("device.jl")

end
