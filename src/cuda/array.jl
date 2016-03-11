const cuda_memstep = 1024
const cuda_ptrs = Dict()

type CudaArray{T,N}
  ptr::Ptr{T}
  dims::NTuple{N,Int}
  dev::Int
end

function allocate!{T}(a::CudaArray{T})
  id = length(a) * sizeof(T) ÷ cuda_memstep + 1
  if haskey(cuda_ptrs, id)
    ptrs = cuda_ptrs[id]
  else
    ptrs = Ptr{Void}[]
    cuda_ptrs[id] = ptrs
  end
  if length(ptrs) == 0
    p = Ptr{Void}[0]
    cudaMalloc(p, id * cuda_memstep)
    a.ptr = p[1]
  else
    a.ptr = convert(Ptr{T}, pop!(ptrs))
  end
end

function release{T}(a::CudaArray{T})
  id = length(a) * sizeof(T) ÷ cuda_memstep + 1
  p = convert(Ptr{Void}, a.ptr)
  push!(cuda_ptrs[id], p)
end

function CudaArray{T,N}(::Type{T}, dims::NTuple{N,Int})
  a = CudaArray(Ptr{T}(0), dims, 1)
  finalizer(a, release)
  a
end
CudaArray{T}(::Type{T}, dims...) = CudaArray(T, dims)

typealias CudaVector{T} CudaArray{T,1}
typealias CudaMatrix{T} CudaArray{T,2}

Base.length(a::CudaArray) = prod(a.dims)
Base.size(a::CudaArray) = a.dims
Base.ndims{T,N}(a::CudaArray{T,N}) = N
Base.eltype{T}(a::CudaArray{T}) = T
Base.stride(a::CudaArray, dim::Int) = prod(size(a)[1:dim-1])
Base.similar{T}(a::CudaArray{T}) = CudaArray(T, size(a))
