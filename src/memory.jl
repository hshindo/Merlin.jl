type MemoryManager
  step::Int
  ptrs::Dict
end

MemoryManager(step=1024) = MemoryManager(step, Dict())

function alloc!{A,T}(mm::MemoryManager, ::Type{A}, ::Type{T}, dims)
  id = prod(dims) * sizeof(T) ÷ memstep + 1
  if haskey(mm.dict, id)
    ptrs = mm.ptrs[id]
  else
    ptrs = Ptr{Void}[]
    mm.ptrs[id] = ptrs
  end
  p = length(ptrs) == 0 ? malloc(T, id*memstep) : pop!(ptrs)
  p = convert(Ptr{T}, p)
end

function release(mm::MemoryManager, x)
  id = length(x) * sizeof(eltype(x)) ÷ memstep + 1
  p = convert(Ptr{Void}, pointer(x))
  push!(mm.ptrs[id], p)
end
