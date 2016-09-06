module CUDNN

using JuCUDA

include("lib/5.1/libcudnn.jl")
include("lib/5.1/libcudnn_types.jl")

if is_windows()
    const libcudnn = Libdl.find_library(["cudnn64_5"])
else
    const libcudnn = Libdl.find_library(["libcudnn"])
end
isempty(libcudnn) && throw("CUDNN library cannot be found.")
info("CUDNN version: $(cudnnGetVersion())")

function checkstatus(status)
    if status != CUDNN_STATUS_SUCCESS
        Base.show_backtrace(STDOUT, backtrace())
        throw(bytestring(cudnnGetErrorString(status)))
    end
end

datatype(::Type{Float32}) = CUDNN_DATA_FLOAT
datatype(::Type{Float64}) = CUDNN_DATA_DOUBLE
datatype(::Type{Float16}) = CUDNN_DATA_HALF

include("handle.jl")
include("tensor.jl")
include("activation.jl")
#include("convolution.jl")
#include("dropout.jl")
#include("filter.jl")
#include("lrn.jl")
#include("pooling.jl")
#include("softmax.jl")
#include("batchnorm.jl")
#include("rnn.jl")

end
