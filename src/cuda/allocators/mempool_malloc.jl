const MEMPOOL = [Cptr[] for i=1:30]

struct MemPoolMalloc
end

function (::MemPoolMalloc)(::Type{T}, dims::Dims{N}) where {T,N}
    bytesize = prod(dims) * sizeof(T)
    @assert bytesize > 0
    bytesize == 0 && return CuArray(Ptr{T}(C_NULL),dims,getdevice())

    c = log2ceil(bytesize) # 2^c >= bytesize
    bytesize = 1 << c
    ptrs = MEMPOOL[c]
    if isempty(ptrs)
        ptr = Ptr{T}(memalloc(bytesize))
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
        ptr = Ptr{T}(pop!(ptrs))
    end
    arr = CuArray(ptr, dims, getdevice())
    function release(x::CuArray{T}) where T
        bytesize = prod(size(x)) * sizeof(T)
        c = log2ceil(bytesize)
        push!(MEMPOOL[c], Cptr(x.ptr))
    end
    finalizer(release, arr)
    arr
end

function dispose(x::MemPoolMalloc)
    gc()
    for dptrs in x.dptrs
        for dptr in dptrs
            memfree(dptr)
        end
    end
end

function log2ceil(bytesize::Int)
    @assert bytesize > 0
    bytesize -= 1
    x = 1
    while bytesize > 1
        bytesize >>= 1
        x += 1
    end
    x
end
