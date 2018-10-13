mutable struct MemPoolMalloc
    size2ptrs::Vector{Vector{Cptr}}
end

function MemPoolMalloc()
    size2ptrs = [Cptr[] for i=1:30]
    MemPoolMalloc(size2ptrs)
end

const MEMPOOL = MemPoolMalloc()

Base.getindex(m::MemPoolMalloc, i::Int) = m.size2ptrs[i]

function (malloc::MemPoolMalloc)(::Type{T}, size::Int) where T
    size == 0 && return CuPtr{T}()
    bytesize = sizeof(T) * size
    id = log2id(bytesize)
    bytesize = 1 << id
    ptrs = malloc[id]
    if isempty(ptrs)
        cptr = memalloc(bytesize)
        #status = @apicall_nocheck :cuMemAlloc (Ptr{UInt64},Csize_t) ref bytesize
        #if status != CUDA_SUCCESS
        #    gc()
        #    if isempty(buffer)
        #        dispose(mem)
        #        dptr = memalloc(bytesize)
        #    else
        #        dptr = pop!(buffer)
        #    end
        #else
        #    dptr = ref[]
        #end
    else
        cptr = pop!(ptrs)
    end
    cuptr = CuPtr(Ptr{T}(cptr), id)
    finalizer(cuptr) do x
        push!(MEMPOOL[x.size], Cptr(x.ptr))
    end
    cuptr
end

function dispose(x::MemPoolMalloc)
    gc()
    for dptrs in x.dptrs
        for dptr in dptrs
            memfree(dptr)
        end
    end
end

function log2id(bytesize::Int)
    bufsize = bytesize - 1
    id = 1
    while bufsize > 1
        bufsize >>= 1
        id += 1
    end
    id
end
