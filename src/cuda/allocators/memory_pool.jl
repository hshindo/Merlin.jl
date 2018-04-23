mutable struct MemoryPool
    dptrs::Vector{Vector{UInt64}}

    function MemoryPool()
        x = new(Vector{UInt64}[])
        finalizer(x, dispose)
        x
    end
end

function dispose(x::MemoryPool)
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

function alloc(mem::MemoryPool, bytesize::Int)
    @assert bytesize >= 0
    #dev = getdevice()
    #if dev < 0
    #    throw("GPU device is not set. Call `setdevice(dev)`.")
    #end
    bytesize == 0 && return MemBlock(UInt64(0),0)
    id = log2id(bytesize)
    bytesize = 1 << id

    while length(mem.dptrs) < id
        push!(mem.dptrs, MemBlock[])
    end
    buffer = mem.dptrs[id]
    if isempty(buffer)
        ref = Ref{UInt64}()
        status = @apicall_nocheck :cuMemAlloc (Ptr{UInt64},Csize_t) ref bytesize
        if status != CUDA_SUCCESS
            gc()
            if isempty(buffer)
                dispose(mem)
                dptr = memalloc(bytesize)
            else
                dptr = pop!(buffer)
            end
        else
            dptr = ref[]
        end
    else
        dptr = pop!(buffer)
    end
    ptr = MemBlock(dptr, bytesize)
    finalizer(ptr, x -> free(mem,x))
    ptr
end

function free(mem::MemoryPool, ptr::MemBlock)
    id = log2id(ptr.bytesize)
    push!(mem.dptrs[id], ptr.dptr)
end
