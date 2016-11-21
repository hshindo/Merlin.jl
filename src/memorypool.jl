type MemoryPool
    array
    index::Int
end

MemoryPool() = MemoryPool(Array{Bool}(2^25), 1)

const mempools = MemoryPool[MemoryPool()]

function alloc(T::Type, dims::Tuple{Vararg{Int}})
    isempty(mempools) && return Array{T}(dims)

    mp = mempools[1]
    len = prod(dims) * sizeof(T)
    @assert len > 0
    @assert len <= length(mp.array)

    count = length(mp.array) - mp.index + 1
    if count < len
        mp.array = similar(mp.array)
        mp.index = 1
        #l = length(mp.array)
        #while l < mp.index+len-1
        #    l *= 2
        #end
        #mp.array = Array(Bool, l)
        #mp.index = 1
    end

    p = pointer(mp.array, mp.index)
    mp.index += len
    unsafe_wrap(Array, Ptr{T}(p), dims)
end
