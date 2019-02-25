module CUDNN

using ..CUDA

macro cudnn(f, args...)
    quote
        status = ccall(($f,CUDA.libcudnn), Cint, $(map(esc,args)...))
        if status != 0
            p = ccall((:cudnnGetErrorString,CUDA.libcudnn), Ptr{UInt8}, (Cint,), status)
            throw(unsafe_string(p))
        end
    end
end

function version()
    @cudnn :cudnnGetVersion ()
end

include("define.jl")
include("handle.jl")

function setstream(handle::Handle, stream)
    @cudnn :cudnnSetStream (Ptr{Cvoid},Ptr{Cvoid}) handle stream
end

include("activation.jl")
include("add.jl")
include("batchnorm.jl")
include("convolution.jl")
include("filter.jl")
include("dropout.jl")
include("pooling.jl")
include("reduce.jl")
include("rnn.jl")
include("softmax.jl")
include("tensor.jl")

end
