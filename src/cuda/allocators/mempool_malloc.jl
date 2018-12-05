const MEMPOOL = [Cptr[] for i=1:30]

struct MemPoolMalloc
end

function (::MemPoolMalloc)(::Type{T}, dims::Dims{N}) where {T,N}
    bytesize = prod(dims) * sizeof(T)
    @assert bytesize > 0

    c = log2ceil(bytesize) # 2^c >= bytesize
    bytesize = 1 << c
    ptrs = MEMPOOL[c]
    if isempty(ptrs)
        ref = Ref{Cptr}()
        status = @unsafe_apicall :cuMemAlloc (Ptr{Cptr},Csize_t) ref bytesize
        if status != CUDA_SUCCESS
            CUDA.synchronize()
            GC.gc()
            if isempty(ptrs)
                for cptrs in MEMPOOL
                    memfree.(cptrs)
                    empty!(cptrs)
                end
                status = @unsafe_apicall :cuMemAlloc (Ptr{Cptr},Csize_t) ref bytesize
                status == CUDA_SUCCESS || throw("Out of Memory from Merlin.")
                ptr = Ptr{T}(ref[])
            else
                ptr = Ptr{T}(pop!(ptrs))
            end
        else
            ptr = Ptr{T}(ref[])
        end
    else
        ptr = Ptr{T}(pop!(ptrs))
    end
    cuptr = CuPtr(ptr, getdevice(), c)
    push!(ALLOCATED, cuptr)
    finalizer(cuptr) do x
        push!(MEMPOOL[c], Cptr(x.ptr))
    end
    cuptr
end

function rrr()
    ref = Ref{Ptr{Cvoid}}()
    status = @unsafe_apicall :cuMemAlloc (Ptr{Ptr{Cvoid}},Csize_t) ref bytesize
    if status != CUDA_SUCCESS
        gc(false)
        status = @unsafe_apicall :cuMemAlloc (Ptr{Ptr{Cvoid}},Csize_t) ref bytesize
        if status != CUDA_SUCCESS
            gc()
            memalloc(bytesize)
        end
    end
    ref[]
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
