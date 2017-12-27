const CUdeviceptr = UInt64
const FreeCuPtrs = [Dict{Int,Vector{CUdeviceptr}}() for i=1:ndevices()]

mutable struct CuPtr
    ptr::CUdeviceptr
    bufid::Int
    dev::Int
end

function CuPtr(bytesize::Int)
    dev = getdevice()
    bufid = (bytesize-1) >> 10 + 1
    bytesize = bufid << 10

    buffers = FreeCuPtrs[dev+1]
    buffer = get!(buffers,bufid) do
        Ptr{CUdeviceptr}[]
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

Base.convert(::Type{Ptr{T}}, p::CuPtr) where T = Ptr{T}(p.ptr)
Base.convert(::Type{CUdeviceptr}, p::CuPtr) = p.ptr
Base.unsafe_convert(::Type{Ptr{T}}, p::CuPtr) where T = Ptr{T}(p.ptr)
Base.unsafe_convert(::Type{CUdeviceptr}, p::CuPtr) = p.ptr

# not tested
function devicegc()
    gc()
    _dev = getdevice()
    for dev = 0:ndevices()-1
        setdevice(dev)
        for (id,ptrs) in freeptrs[dev+1]
            for p in ptrs
                cuMemFree(p)
            end
            empty!(ptrs)
        end
    end
    setdevice(_dev)
    run(`nvidia-smi`)
end
