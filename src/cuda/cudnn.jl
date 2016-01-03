export CUDNN
module CUDNN

using CUDArt

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
  const libcudnn = Libdl.find_library(["cudnn64_4.dll"])
end : begin
  const libcudnn = Libdl.find_library(["libcudnn"])
end)
!isempty(libcudnn) || error("CUDNN library cannot be found.")

macro cudnncall(f, argtypes, args...)
  quote
    status = ccall(($f, $libcudnn), Cint, $argtypes, $(args...))
    if status != STATUS_SUCCESS
      str = bytestring(ccall((:cudnnGetErrorString, $libcudnn), Cstring, (Cint,), status))
      throw(str)
    end
  end
end

##### handle #####
const handles = Dict{Int, Ptr{Void}}()
atexit(() -> for h in handles destroy(h) end)

function gethandle(dev::Int)
  if !haskey(handles, dev)
    h = Ptr{Void}[0]
    @cudnncall(:cudnnCreate, (Ptr{Ptr{Void}},), h)
    handles[dev] = h[1]
    h[1]
  else
    handles[dev]
  end
end

function destroy(handle)
  @cudnncall(:cudnnDestroy, (Ptr{Void},), handle)
end

function setstream(handle, stream)
  @cudnncall(:cudnnSetStream, (Ptr{Void},Ptr{Void}), handle, stream)
end

function getstream(handle)
  s_handle = Ptr{Void}[0]
  @cudnncall(:cudnnGetStream, (Ptr{Void},Ptr{Ptr{Void}}), handle, s_handle)
  return s_handle[1]
end

datatype(::AbstractCudaArray{Float32}) = 0 # CUDNN_DATA_FLOAT
datatype(::AbstractCudaArray{Float64}) = 1 # CUDNN_DATA_DOUBLE
datatype(::AbstractCudaArray{Float16}) = 2 # CUDNN_DATA_HALF

destroy_tensor_descriptor(desc) = @cudnncall(:cudnnDestroyTensorDescriptor, (Ptr{Void},), desc)

function create_tensor_descriptor(a::AbstractCudaArray)
  csize = Cint[reverse(size(a))...]
  cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
  d = Ptr{Void}[0]
  @cudnncall(:cudnnCreateTensorDescriptor, (Ptr{Void},), d)
  desc = d[1]
  @cudnncall(:cudnnSetTensorNdDescriptor, (Ptr{Void},Cint,Cint,Ptr{Cint},Ptr{Cint}),
    desc, datatype(a), ndims(a), csize, cstrides)
  #finalizer(desc, destroy_tensor_descriptor)
  desc
end

""" y = alpha * x + beta * y """
function add{T}(x::AbstractCudaArray{T}, y::AbstractCudaArray{T}; alpha=1.0, beta=0.0)
  handle = gethandle(bias.dev)
  xdesc = descriptor(x)
  ydesc = descriptor(y)
  @cudnncall(:cudnnAddTensor, (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
    handle, T[alpha], xdesc, x, T[beta], ydesc, y)
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
  @cudnncall(:cudnnActivationForward, (Ptr{Void},Cint,Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
    handle, mode, T[alpha], xdesc, x, T[beta], ydesc, y)
end

end
