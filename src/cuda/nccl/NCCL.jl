module NCCL

using ..CUDA
import Libdl

if Sys.iswindows()
    const libnccl = ""
else
    const libnccl = Libdl.find_library("libnccl")
end
isempty(libnccl) && @warn "NCCL cannot be found."

function init()
    @info "NCCL API xxxx"
    # global const API_VERSION = Int(ccall((:cudnnGetVersion,libcudnn),Cint,()))
    # @info("NCCL API $API_VERSION")
end
init()

macro nccl(f, args...)
    quote
        result = ccall(($f,libnccl), Cint, $(map(esc,args)...))
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
