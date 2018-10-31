module CUDNN

using ..CUDA
import ..CUDA: ndevices, getdevice
import Libdl

if Sys.iswindows()
    const libcudnn = Libdl.find_library(["cudnn64_7"])
else
    const libcudnn = Libdl.find_library("libcudnn")
end
isempty(libcudnn) && error("CUDNN cannot be found.")

const API_VERSION = Ref{Int}()

function __init__()
    API_VERSION[] = Int(ccall((:cudnnGetVersion,libcudnn),Cint,()))
    @info "CUDNN API $(API_VERSION[])"
end

macro cudnn(f, args...)
    quote
        status = ccall(($f,libcudnn), Cint, $(map(esc,args)...))
        if status != 0
            p = ccall((:cudnnGetErrorString,libcudnn), Ptr{UInt8}, (Cint,), status)
            throw(unsafe_string(p))
        end
    end
end

include("define.jl")
include("handle.jl")

function setstream(handle::Handle, stream)
    @cudnn :cudnnSetStream (Ptr{Cvoid},Ptr{Cvoid}) handle stream
end

const ALLOCATED = []

include("activation.jl")
include("add.jl")
include("convolution.jl")
include("filter.jl")
include("dropout.jl")
include("reduce.jl")
include("rnn.jl")
include("softmax.jl")
include("tensor.jl")

end
