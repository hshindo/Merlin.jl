type CudaArray{T,N}
  ptr::Ptr{T}
  dims::NTuple{N,Int}
  dev::Int
end

function CudaArray(T::Type, dims::Dims)
  p = get_ptr!(CudaArray, T, dims)
  a = CudaArray(p, dims, device())
  finalizer(a, release)
  a
end

typealias CudaVector{T} CudaArray{T,1}
typealias CudaMatrix{T} CudaArray{T,2}

Base.length(a::CudaArray) = prod(a.dims)
Base.pointer(a::CudaArray) = a.ptr
Base.size(a::CudaArray) = a.dims
Base.ndims{T,N}(a::CudaArray{T,N}) = N
Base.eltype{T}(a::CudaArray{T}) = T
Base.stride(a::CudaArray, dim::Int) = prod(size(a)[1:dim-1])
device(a::CudaArray) = a.dev

function CudaArray(T::Type, dims::Dims)
  n = prod(dims)
  p = Ptr{Void}[0]
  cudaMalloc(p, n)
  p = convert(Ptr{T}, p[1])
  CudaArray(p, dims, 1)
end
CudaArray(T::Type, dims::Int...) = CudaArray(T, dims)
