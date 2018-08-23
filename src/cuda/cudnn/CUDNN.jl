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

include("activation.jl")
include("convolution.jl")
include("filter.jl")
include("dropout.jl")
include("reduce.jl")
include("rnn.jl")
include("softmax.jl")
include("tensor.jl")

"""
C = α*A + β*C

The bias tensor A must match the corresponding dimension of the destination tensor
C or must be equal to 1.
"""
function addto!(α, A::CuArray{T}, β, C::CuArray{T}) where T
    h = gethandle()
    adesc = TensorDesc(A, 4)
    cdesc = TensorDesc(C, 4)
    @cudnn(:cudnnAddTensor,
        (Cptr,Cptr,Cptr,Cptr,Cptr,Cptr,Cptr),
        h, T[α], adesc, A, T[β], cdesc, C)
    C
end

end
