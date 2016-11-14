type MemoryPool
    array
    index::Int
end

function MemoryPool()
    array = Array(Bool, 16)
    MemoryPool(array, 1)
end

function alloc!(mp::MemoryPool, T::Type, dims::Tuple)
    len = prod(dims) * sizeof(T)
    @assert len > 0

    count = length(mp.array) - mp.index + 1
    if count < len
        l = length(mp.array)
        while l < mp.index+len-1
            l *= 2
        end
        mp.array = Array(Bool, l)
        mp.index = 1
    end

    p = pointer(mp.array, mp.index)
    mp.index += len
    unsafe_wrap(Array, Ptr{T}(p), dims)
end
