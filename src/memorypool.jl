"""
Memory Pool
Memory pool maintains CPU and GPU memories.
"""
type MemoryPool
  pool::Vector
end

function allocate(mp::MemoryPool, dims::Int...)
  l = prod(dims)
  buffer = mp.pool[end]
  if mp.index + l > length(buffer)
    a = pop!(FreeMemorySet)
    push!(a, mp.pool)
  end
  pointer_to_array(pointer(a), dims)
end

function free(mp::MemoryPool)
  for a in mp.pool
    push!(FreeMemorySet, a)
  end
end

const MemorySet = []
const FreeMemorySet = []

const MMMM = begin
  d = ObjectIdDict()
  for i = 1:16
    a = Array(Float32, 2 << 20)
    d[a] = true
  end
end

type MemoryPool
  buffer::Vector
  index::Int
end

function alloc(mp::MemoryPool, dims::Int...)
  l = prod(dims)
  pl = length(mp.buffer[1]) - mp.indices[end]
  if l > pl
    push!(mp.buffer, )
  end
  pointer_to_array(pointer(a), dims)
end
