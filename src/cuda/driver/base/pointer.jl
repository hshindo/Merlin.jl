const freeptrs = [Dict{Int,Vector{CUdeviceptr}}() for i=1:devcount()]

type CuPtr{T}
    ptr::CUdeviceptr
    bytelen::Int
    dev::Int

    function CuPtr(ptr, bytelen, dev)
        p = new(ptr, bytelen, dev)
        finalizer(p, free)
        p
    end
end

Base.convert{T}(::Type{Ptr{T}}, p::CuPtr) = Ptr{T}(p.ptr)
Base.convert(::Type{CUdeviceptr}, p::CuPtr) = p.ptr
Base.unsafe_convert{T}(::Type{Ptr{T}}, p::CuPtr) = Ptr{T}(p.ptr)
Base.unsafe_convert(::Type{CUdeviceptr}, p::CuPtr) = p.ptr

function alloc{T}(::Type{T}, len::Int)
    dev = device()
    bytelen = (sizeof(T)*len - 1) >> 10
    bytelen = (bytelen+1) << 10

    ptrs = freeptrs[dev+1]
    if haskey(ptrs,bytelen) && length(ptrs[bytelen]) > 0
        ptr = pop!(ptrs[bytelen])
    else
        ptrs[bytelen] = CUdeviceptr[]
        ref = CUdeviceptr[0]
        cuMemAlloc(ref, bytelen)
        ptr = ref[1]
    end
    CuPtr{T}(ptr, bytelen, dev)
end

free(p::CuPtr) = push!(freeptrs[p.dev+1][p.bytelen], p.ptr)

# not tested
function devicegc()
    gc()
    _dev = device()
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
