#const memstep = 1024
const cuda_ptrs = Dict()

function alloc_cuda{T}(::Type{T}, dims)
  id = prod(dims) * sizeof(T) ÷ memstep + 1
  if haskey(cuda_ptrs, id)
    ptrs = cuda_ptrs[id]
  else
    ptrs = Ptr{Void}[]
    cpu_ptrs[id] = ptrs
  end
  p = length(ptrs) == 0 ? CUDArt.malloc(T, id*memstep) : pop!(ptrs)
  a = CudaArray(convert(Ptr{T}, p), dims, device())
  finalizer(a, release_cuda)
  a
end
alloc_cuda{T}(::Type{T}, dims::Int...) = alloc_cuda(T, dims)
alloc_cuda{T}(a::CudaArray{T}) = alloc_cuda(T, size(a))
