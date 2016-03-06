const cpu_mem = MemoryManager()

malloc{T<:Array}(::Type{T}, size::Int) = Libc.malloc(size)

function Array{T,N}(::Type{T}, dims::NTuple{N,Int})
  p = alloc!(cpu_mem, prod(dims))
  pointer_to_array(p, dims)
end
