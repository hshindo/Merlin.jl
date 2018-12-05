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
