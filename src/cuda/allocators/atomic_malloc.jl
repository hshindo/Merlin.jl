mutable struct AtomicMalloc
    blocksize::Int
    ptrs::Vector{Cptr}
    blockid::Int
    offset::Int

    function AtomicMalloc(blocksize::Int=1024*1000*1000)
        m = new(blocksize, Cptr[], 0, 0)
        # finalizer(m, dispose)
        m
    end
end

const ATOMIC_MALLOC = AtomicMalloc()

function (m::AtomicMalloc)(::Type{T}, dims::Dims{N}) where {T,N}
    bytesize = prod(dims) * sizeof(T)
    @assert 0 < bytesize <= m.blocksize

    if m.blockid == 0 || bytesize + m.offset > m.blocksize
        if m.blockid == length(m.ptrs)
            push!(m.ptrs, memalloc(m.blocksize))
        end
        m.blockid += 1
        ptr = m.ptrs[m.blockid]
        m.offset = 0
    else
        ptr = m.ptrs[m.blockid]
    end
    m.offset += bytesize
    CuArray(Ptr{T}(ptr), dims, getdevice())
end

function reset(m::AtomicMalloc)
    m.blockid = 1
    m.offset = 0
end

function dispose(m::AtomicMalloc)
    setdevice(m.device) do
        foreach(memfree, m.ptrs)
    end
end
