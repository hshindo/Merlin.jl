module NCCL

using ..CUDA

macro nccl(f, args...)
    quote
        result = ccall(($f,CUDA.libnccl), Cint, $(map(esc,args)...))
        if result != 0
            p = ccall((:ncclGetErrorString,libnccl), Ptr{UInt8}, (Cint,), result)
            throw(unsafe_string(p))
        end
    end
end

include("types.jl")
include("comm.jl")
include("collective.jl")

end
