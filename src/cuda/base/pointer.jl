const freeptrs = [Dict{Int,Vector{CUdeviceptr}}() for i=1:devcount()]

type CuPtr
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

    dict = freeptrs[dev+1]
    if haskey(dict,bytelen) && length(dict[bytelen]) > 0
        ptr = pop!(dict[bytelen])
    else
        dict[bytelen] = Ptr{Void}[]
        p = CUdeviceptr[0]
        cuMemAlloc(p, bytelen)
        ptr = p[1]
    end
    CuPtr(ptr, bytelen, dev)
end

free(p::CuPtr) = push!(freeptrs[p.dev+1][p.bytelen], p.ptr)

# not tested
function devicegc()
    gc()
    _dev = device()
    for dev = 0:devcount()-1
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
