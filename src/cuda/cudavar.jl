using CUDArt

type CudaVar{T,N}
  value::CudaArray{T,N}
  fixed::Bool
  buffer::CudaVector{T}
end

CudaVar{T,N}(value::CudaArray{T,N}) = CudaVar(value, false, CudaArray(T, 1))

default{T,N}(v::CudaVar{T,N}) = CudaVar(similar(v.value))

function setvalue!{T,N}(v::CudaVar{T,N}, dims::NTuple{N,Int})
  dims == size(v.value) && return
  len = prod(dims)
  len <= 0 && error("length <= 0")
  if length(v.buffer) < len
    k = length(v.buffer)
    while k < len
      k *= 2
    end
    v.buffer = CudaArray(T, k)
  end
  v.value.ptr = v.buffer.ptr
  v.value.dims = dims
  v.value.dev = v.buffer.dev
end
