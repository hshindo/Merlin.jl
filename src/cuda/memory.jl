mutable struct MemoryBuffer
    index::Int
    bytesize::Int
    ptr::UInt64
    dev::Int
end
function MemoryBuffer()
    dev = getdevice()
    MemoryBuffer(0, )
end

const MemBuffers = MemoryBuffer[]

function malloc!(mem::MemoryBuffer, ::Type{T}, dims::NTuple{N,Int}) where {T,N}
    bytesize = sizeof(T) * prod(dims)
    if bytesize + mem.index <= mem.bytesize
        ptr = Ptr{T}(ptr + bytesize)
        mem.index += bytesize
    else
        ref = Ref{Ptr{Void}}()
        @apicall :cuMemAlloc (Ptr{Ptr{Void}},Csize_t) ref bytesize
        ptr = ref[]
    end
    ptr
end

function CuPtr(bytesize::Int)
    @assert bytesize >= 0
    dev = getdevice()
    if dev < 0
        throw("GPU device is not set. Call `setdevice(dev)`.")
    end
    bytesize == 0 && return CuPtr(zero(UInt64),-1,dev)
    bufid = (bytesize-1) >> 10 + 1
    bytesize = bufid << 10

    while length(FreeCuPtrs) < dev + 1
        push!(FreeCuPtrs, Dict{Int,Vector{UInt64}}())
    end
    buffers = FreeCuPtrs[dev+1]
    buffer = get!(buffers,bufid) do
        Ptr{UInt64}[]
    end
    if isempty(buffer)
        ref = Ref{Ptr{Void}}()
        @apicall :cuMemAlloc (Ptr{Ptr{Void}},Csize_t) ref bytesize
        ptr = ref[]
    else
        ptr = pop!(buffer)
    end
    cuptr = CuPtr(ptr, bufid, dev)
    finalizer(cuptr, x -> push!(FreeCuPtrs[x.dev+1][x.bufid], x.ptr))
    cuptr
end
