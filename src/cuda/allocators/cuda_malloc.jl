struct CUDAMalloc
end

function (::CUDAMalloc)(::Type{T}, size::Int) where T
    @assert size >= 0
    size == 0 && return CuPtr{T}()
    ptr = Ptr{T}(memalloc(sizeof(T)*size))
    cuptr = CuPtr(ptr)
    finalizer(x -> memfree(x.ptr), cuptr)
    cuptr
end

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
