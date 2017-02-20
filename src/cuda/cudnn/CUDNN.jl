module CUDNN

using ..CUDA
import ..CUDA: redim

if is_windows()
    const libcudnn = Libdl.find_library(["cudnn64_5","cudnn64_4"])
else
    const libcudnn = Libdl.find_library(["libcudnn"])
end
isempty(libcudnn) && throw("CUDNN library cannot be found.")

const version = ccall((:cudnnGetVersion,libcudnn),Cint,())
const major = div(version, 1000)
const minor = div(version - major*1000, 100)

info("CUDNN version: $(version)")
include("../lib/$(CUDA.major).$(CUDA.minor)/libcudnn$(major)$(minor).jl")
include("../lib/$(CUDA.major).$(CUDA.minor)/libcudnn$(major)$(minor)_types.jl")

function checkstatus(status)
    status == CUDNN_STATUS_SUCCESS && return
    Base.show_backtrace(STDOUT, backtrace())
    throw(bytestring(cudnnGetErrorString(status)))
end

datatype(::Type{Float32}) = CUDNN_DATA_FLOAT
datatype(::Type{Float64}) = CUDNN_DATA_DOUBLE
datatype(::Type{Float16}) = CUDNN_DATA_HALF

const handles = Ptr{Void}[]
function handle(x)
    dev = device(x) + 1
    while dev > length(handles)
        p = Ptr{Void}[0]
        cudnnCreate(p)
        push!(handles, p[1])
    end
    handles[dev]
end
atexit(() -> foreach(cudnnDestroy, handles))

include("tensor.jl")
include("activation.jl")
#include("batchnorm.jl")
#include("convolution.jl")
#include("dropout.jl")
##include("lrn.jl")
#include("pooling.jl")
include("softmax.jl")
##include("rnn.jl")

end
