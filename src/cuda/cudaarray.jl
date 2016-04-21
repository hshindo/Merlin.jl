export CudaArray, CudaVector, CudaMatrix

const cuda_memstep = 1024
const cuda_ptrs = Dict{Int,Vector{Ptr{Void}}}()

type CudaArray{T,N}
  ptr::Ptr{T}
  dims::NTuple{N,Int}
  dev::Int
end
typealias CudaVector{T} CudaArray{T,1}
typealias CudaMatrix{T} CudaArray{T,2}

function CudaArray{T,N}(::Type{T}, dims::NTuple{N,Int}, dev=1)
  id = prod(dims) * sizeof(T) ÷ cuda_memstep + 1
  if haskey(cuda_ptrs, id)
    ptrs = cuda_ptrs[id]
  else
    ptrs = Ptr{Void}[]
    cuda_ptrs[id] = ptrs
  end
  if length(ptrs) == 0
    p = Ptr{Void}[0]
    RT.cudaMalloc(p, id * cuda_memstep)
    ptr = p[1]
  else
    ptr = pop!(ptrs)
  end

  a = CudaArray(convert(Ptr{T},ptr), dims, dev)
  finalizer(a, release)
  a
end
CudaArray{T}(::Type{T}, dims::Int...) = CudaArray(T, dims)
function CudaArray{T,N}(src::Array{T,N}, dev=1)
  a = CudaArray(T, size(src), dev)
  copy!(a, src)
  a
end

function release{T}(a::CudaArray{T})
  id = length(a) * sizeof(T) ÷ cuda_memstep + 1
  p = convert(Ptr{Void}, a.ptr)
  push!(cuda_ptrs[id], p)
end

Base.length(a::CudaArray) = prod(a.dims)
Base.size(a::CudaArray) = a.dims
Base.size(a::CudaArray, d::Int) = a.dims[d]
Base.ndims{T,N}(a::CudaArray{T,N}) = N
Base.eltype{T}(a::CudaArray{T}) = T
Base.stride(a::CudaArray, dim::Int) = prod(size(a)[1:dim-1])
Base.pointer(a::CudaArray) = a.ptr
Base.similar{T}(a::CudaArray{T}) = CudaArray(T, size(a))
Base.unsafe_convert(::Type{Ptr{Void}}, a::CudaArray) = convert(Ptr{Void}, a.ptr)
Base.unsafe_convert{T}(::Type{Ptr{T}}, a::CudaArray{T}) = a.ptr
Base.zeros{T}(a::CudaArray{T}) = CudaArray(T, a.dims, a.dev)
Base.similar{T}(a::CudaArray{T}) = Array(T, a.dims, a.dev)
Base.copy!(dest::Array, src::CudaArray; stream=RT.nullstream) = cuda_copy!(dest, src, RT.cudaMemcpyDeviceToHost, stream)
Base.copy!(dest::CudaArray, src::Array; stream=RT.nullstream) = cuda_copy!(dest, src, RT.cudaMemcpyHostToDevice, stream)
Base.copy!(dest::CudaArray, src::CudaArray; stream=RT.nullstream) = cuda_copy!(dest, src, RT.cudaMemcpyDeviceToDevice, stream)
function Base.Array{T,N}(src::CudaArray{T,N})
  a = Array(T, size(src))
  copy!(a, src)
  a
end

function cuda_copy!(dest, src, kind, stream)
  if length(dest) != length(src)
    throw(ArgumentError("Inconsistent array length."))
  end
  count = length(src) * sizeof(eltype(src))
  RT.cudaMemcpyAsync(dest, src, count, kind, stream)
  dest
end

#=
function device_gc()
  for ptrs in cuda_ptrs
    for p in ptrs
      RT.cudaFree(p)
    end
  end
end
=#
