module CUDNN

using CUJulia

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
include("lib/$(CUJulia.major).$(CUJulia.minor)/libcudnn$(major)$(minor).jl")
include("lib/$(CUJulia.major).$(CUJulia.minor)/libcudnn$(major)$(minor)_types.jl")

function checkstatus(status)
    status == CUDNN_STATUS_SUCCESS && return
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

export
    cudnnActivationForward, cudnnActivationBackward,
    cudnnSoftmaxForward, cudnnSoftmaxBackward

export CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU
export CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL # mode
export CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG # algorithm

type TensorDesc
    ptr::Ptr{Void}
end

function TensorDesc{T,N}(x::CuArray{T,N}; pad=0)
    @assert N <= 4
    csize = Cint[1, 1, 1, 1]
    cstrides = Cint[1, 1, 1, 1]
    st = strides(x)
    for i = 1:N
        csize[4-i-pad+1] = size(x,i)
        cstrides[4-i-pad+1] = st[i]
    end
    p = Ptr{Void}[0]
    cudnnCreateTensorDescriptor(p)
    cudnnSetTensorNdDescriptor(p[1], datatype(T), 4, csize, cstrides)
    desc = TensorDesc(p[1])
    finalizer(desc, cudnnDestroyTensorDescriptor)
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::TensorDesc) = desc.ptr

type ActivationDesc
    ptr::Ptr{Void}
end

function ActivationDesc(mode::UInt32; relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
    p = Ptr{Void}[0]
    cudnnCreateActivationDescriptor(p)
    cudnnSetActivationDescriptor(p[1], mode, relu_nanopt, relu_ceiling)
    desc = ActivationDesc(p[1])
    finalizer(desc, cudnnDestroyActivationDescriptor)
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr

#include("batchnorm.jl")
#include("convolution.jl")
#include("dropout.jl")
##include("lrn.jl")
#include("pooling.jl")
#include("softmax.jl")
##include("rnn.jl")

end
