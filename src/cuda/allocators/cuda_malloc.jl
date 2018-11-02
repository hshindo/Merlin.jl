const ALLOCATED = []

struct CUDAMalloc
end

function (::CUDAMalloc)(::Type{T}, dims::Dims{N}) where {T,N}
    bytesize = sizeof(T) * prod(dims)
    @assert bytesize > 0
    ptr = Ptr{T}(memalloc(bytesize))
    arr = CuArray(ptr, dims, getdevice())
    push!(ALLOCATED, arr)
    finalizer(x -> memfree(x.ptr), arr)
    arr
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
