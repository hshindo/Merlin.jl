mutable struct AtomicMalloc
    blocksize::Int
    device::Int
    ptrs::Vector{Ptr{Void}}
    index::Int
    offset::Int

    function AtomicMalloc(blocksize::Int=1024*1000*1000)
        m = new(blocksize, getdevice(), Ptr{Void}[], 0, 0)
        # finalizer(m, dispose)
        m
    end
end

function (m::AtomicMalloc)(bytesize::Int)
    @assert bytesize <= m.blocksize
    if m.index == 0 || bytesize + m.offset > m.blocksize
        if m.index == length(m.ptrs)
            ptr = memalloc(m.blocksize)
            push!(m.ptrs, ptr)
        end
        ptr = m.ptrs[m.index+1]
        m.index += 1
        m.offset = 0
    end
    m.offset += bytesize
    MemBlock(ptr, bytesize)
end

function reset(m::AtomicMalloc)
    m.index = 1
    m.offset = 0
end

function dispose(m::AtomicMalloc)
    setdevice(m.device) do
        foreach(memfree, m.ptrs)
    end
end
