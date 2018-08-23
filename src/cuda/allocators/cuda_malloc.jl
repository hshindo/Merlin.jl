struct CUDAMalloc
end

function (::CUDAMalloc)(::Type{T}, size::Int) where T
    ptr = memalloc(T, size)
    finalizer(memfree, ptr)
    ptr
end

free(::CUDAMalloc, x::CuPtr) = memfree(x.ptr)

function sss()
    ref = Ref{Ptr{Cvoid}}()
    @apicall :cuMemAlloc (Ptr{Ptr{Cvoid}},Csize_t) ref n*sizeof(T)
    dev = getdevice()
    mb = MemBlock(Ptr{T}(ref[]), n, dev)
    finalizer(mb, memfree)
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
