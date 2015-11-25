using CUDArt

type MemoryPool
  data::CudaVector{Float32}
  offset::Int
end

MemoryPool() = MemoryPool(CudaArray(Float32, 1), 1)

const mempools = Dict{Int, MemoryPool}()

function alloc{T}(mp::MemoryPool, ::Type{T}, dims...)
  len = prod(dims)
  len > 0 || error("invalid dims: $(dims)")
  if mp.offset + len - 1 > length(mp.data)
    n = trunc(Int, log2(mp.offset + len - 1))
    mp.data = CudaArray(T, 2 << n)
    mp.offset = 1
  end
  a = CudaArray(mp.data.ptr, dims, mp.data.dev)
  #a = pointer_to_array(pointer(p.data, p.offset), dims)
  mp.offset += len
  a
end

free(mp) = mp.offset = 1
