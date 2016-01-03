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
!isempty(libcudnn) || error("CuDNN library cannot be found")

macro cudnncall(f, argtypes, args...)
  quote
    status = ccall(($f, $libcudnn), Cint, $argtypes, $(args...))
    if status != STATUS_SUCCESS
      str = bytestring(ccall((:cudnnGetErrorString, $libcudnn), Cstring, (Cint,), status))
      throw(str)
    end
  end
end

type Descriptor
  ptr
end

########## handle ##########
const handles = Dict{Int, Ptr{Void}}()
atexit(() -> for h in handles destroy(h) end)

function create()
  dev = device()
  if !haskey(handles, dev)
    h = Ptr{Void}[0]
    @cudnncall(:cudnnCreate, (Ptr{Ptr{Void}},), h)
    handles[device()] = h[1]
    h[1]
  else
    handles[dev]
  end
end

function destroy(handle)
  @cudnncall(:cudnnDestroy, (Ptr{Void},), handle)
end

function set_stream(handle, stream)
  @cudnncall(:cudnnSetStream, (Ptr{Void},Ptr{Void}), handle, stream)
end

function get_stream(handle)
  s_handle = Ptr{Void}[0]
  @cudnncall(:cudnnGetStream, (Ptr{Void},Ptr{Ptr{Void}}), handle, s_handle)
  return s_handle[1]
end

datatype(::AbstractCudaArray{Float32}) = 0 # CUDNN_DATA_FLOAT
datatype(::AbstractCudaArray{Float64}) = 1 # CUDNN_DATA_DOUBLE
datatype(::AbstractCudaArray{Float16}) = 2 # CUDNN_DATA_HALF

type CuArray
  ptr::Ptr{Void}
  desc::Ptr{Void}
end
Base.show(io::IO, desc::Descriptor) = print(io, desc.ptr)

function create_tensor_descriptor(a::AbstractCudaArray)
  revsize = Cint[reverse(size(a))...]
  revstrides = Cint[stride(a, i) for i=ndims(a):-1:1]
  d = Ptr{Void}[0]
  @cudnncall(:cudnnCreateTensorDescriptor, (Ptr{Void},), d)
  desc = Descriptor(d[1])
  @cudnncall(:cudnnSetTensorNdDescriptor, (Ptr{Void},Cint,Cint,Ptr{Cint},Ptr{Cint}),
    desc.ptr, datatype(a), length(revsize), revsize, revstrides)
  #finalizer(desc, free)
  desc
end

# dest = alpha * bias + beta * dest
function add_tensor{T}(handle, alpha, bias_desc, bias::AbstractCudaArray{T}, beta, dest_desc, dest)
  @cudnncall(:cudnnAddTensor_v3, (Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void},Ptr{Void}),
  handle, T[alpha], bias_desc.ptr, bias, T[beta], dest_desc.ptr, dest)
end

# src .= value
#function set_tensor(handle, src::AbstractCudaArray, value::Number)
#  @cudnncall(:cudnnSetTensor, (Handle,TensorDescriptor,), handle)
#    cudnnSetTensor(handle, TD(src,4), src, cptr(value,src))
#    return src
#end

end
