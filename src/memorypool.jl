function log2int(x::Int)
  k = 1
  while (2 << k) < x
    k += 1
  end
  k
end

const ALL = begin
  buffer = Vector{Array{Float32}}[]
  for i = 1:20
    push!(buffer, Array{Float32}[])
  end
  buffer
end

const MEMPOOL = begin
  buffer = Vector{Array{Float32}}[]
  for i = 1:20
    push!(buffer, Array{Float32}[])
  end
  buffer
end

function malloc(dims::Int...)
  count = prod(dims)
  count > 0 || error("invalid dims: $(dims)")
  k = log2int(count)
  pool = MEMPOOL[k]
  if length(pool) == 0
    a = Array(Float32, 2 << k)
    push!(pool, a)
    push!(ALL[k], a)
  end
  a = pop!(pool)
  pointer_to_array(pointer(a), dims)
end

function free(a::Array)
  k = log2int(length(a))
  push!(MEMPOOL[k], a)
  nothing
end
