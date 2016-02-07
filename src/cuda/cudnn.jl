export CUDNN
module CUDNN

using CUDArt

include("libcudnn.jl")

const STATUS_SUCCESS = 0
const STATUS_NOT_INITIALIZED = 1
const STATUS_ALLOC_FAILED = 2
const STATUS_BAD_PARAM = 3
const STATUS_INTERNAL_ERROR = 4
const STATUS_INVALID_VALUE = 5
const STATUS_ARCH_MISMATCH = 6
const STATUS_MAPPING_ERROR = 7
const STATUS_EXECUTION_FAILED = 8
const STATUS_NOT_SUPPORTED = 9
const STATUS_LICENSE_ERROR = 10

@windows? (
begin
  const libcudnn = Libdl.find_library(["cudnn64_4"])
end : begin
  const libcudnn = Libdl.find_library(["libcudnn"])
end)
# !isempty(libcudnn) || error("CUDNN library cannot be found.")

##### Handle #####
const handles = Dict{Int, Ptr{Void}}()
atexit(() -> for h in handles destroy(h) end)

function gethandle(dev::Int)
  if !haskey(handles, dev)
    h = Ptr{Void}[0]
    cudnnCreate(h)
    handles[dev] = h[1]
    h[1]
  else
    handles[dev]
  end
end

datatype(::AbstractCudaArray{Float32}) = 0 # CUDNN_DATA_FLOAT
datatype(::AbstractCudaArray{Float64}) = 1 # CUDNN_DATA_DOUBLE
datatype(::AbstractCudaArray{Float16}) = 2 # CUDNN_DATA_HALF

##### Descriptor #####
function create_tensor_descriptor(a::AbstractCudaArray)
  csize = Cint[size(a,i) for i=ndims(a):-1:1]
  cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
  d = Ptr{Void}[0]
  cudnnCreateTensorDescriptor(d)
  desc = d[1]
  cudnnSetTensorNdDescriptor(desc, datatype(a), ndims(a), csize, cstrides)
  #finalizer(desc, cudnnDestroyTensorDescriptor
  desc
end

""" y = alpha * x + beta * y """
function add{T}(x::AbstractCudaArray{T}, y::AbstractCudaArray{T}; alpha=1.0, beta=0.0)
  handle = gethandle(bias.dev)
  xdesc = descriptor(x)
  ydesc = descriptor(y)
  #@cudnncall(:cudnnAddTensor, (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
  #  handle, T[alpha], xdesc, x, T[beta], ydesc, y)
end

##### Activation #####
const ACTIVATION_SIGMOID = 0
const ACTIVATION_RELU = 1
const ACTIVATION_TANH = 2
const ACTIVATION_CLIPPED_RELU = 3

function activation_forward{T}(mode::Int, x::AbstractCudaArray{T}, y::AbstractCudaArray{T}; alpha=1.0, beta=0.0)
  handle = gethandle(x.dev)
  xdesc = create_tensor_descriptor(x)
  ydesc = create_tensor_descriptor(y)
  cudnnActivationForward(handle, mode, T[alpha], xdesc, x, T[beta], ydesc, y)
end

function activation_forward2{T}(mode::Int, x::AbstractCudaArray{T}, xdesc, y::AbstractCudaArray{T}, ydesc; alpha=1.0, beta=0.0)
  handle = gethandle(x.dev)
  #xdesc = create_tensor_descriptor(x)
  #ydesc = create_tensor_descriptor(y)
  cudnnActivationForward(handle, mode, T[alpha], xdesc, x, T[beta], ydesc, y)
end

function activation_backward{T}(mode::Int, x::AbstractCudaArray{T}, dx, y::AbstractCudaArray{T}, dy; alpha=1.0, beta=0.0)
  handle = gethandle(x.dev)
  xdesc = create_tensor_descriptor(x)
  dxdesc = create_tensor_descriptor(dx)
  ydesc = create_tensor_descriptor(y)
  dydesc = create_tensor_descriptor(dy)
  cudnnActivationBackward(handle, mode, T[alpha], ydesc, y, dydesc, dy, xdesc, x, T[beta], dxdesc, dx)
end

##### Convolution #####


end
