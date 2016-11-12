export MemoryPool

type MemoryPool
    array::Array
    length::Int
    index::Int
end

function MemoryPool()
    a = Array(Bool, 16)
    MemoryPool(a, 16, 1)
end

function Base.Array(mp::MemoryPool, T::Type, dims::Tuple{Vararg{Int}})
    len = prod(dims) * sizeof(T)
    @assert len > 0

    count = mp.length - mp.index + 1
    if count < len
        while mp.length < mp.index+len-1
            mp.length *= 2
        end
        mp.array = Array(Bool, mp.length)
        mp.index = 1
    end

    p = pointer(mp.array, mp.index)
    mp.index += len
    unsafe_wrap(Array, Ptr{T}(p), dims)
end

Base.similar{T}(mp::MemoryPool, x::Array{T}) = Array(mp, T, size(x))
