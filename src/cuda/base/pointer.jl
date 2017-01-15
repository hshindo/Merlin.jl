const freeptrs = [Dict{Int,Vector{Ptr{Void}}}() for i=1:devcount()]

type CudaPtr{T}
    ptr::Ptr{T}
    bytelen::Int
    dev::Int

    function CudaPtr(ptr, bytelen, dev)
        p = new(ptr, bytelen, dev)
        finalizer(p, free)
        p
    end
end

Base.convert{T}(::Type{Ptr{T}}, p::CudaPtr) = Ptr{T}(p.ptr)
Base.unsafe_convert{T}(::Type{Ptr{T}}, p::CudaPtr) = Ptr{T}(p.ptr)

function alloc{T}(::Type{T}, len::Int)
    dev = device()
    bytelen = (sizeof(T)*len - 1) >> 10
    bytelen = (bytelen+1) << 10

    dict = freeptrs[dev+1]
    if haskey(dict,bytelen) && length(dict[bytelen]) > 0
        ptr = pop!(dict[bytelen])
    else
        dict[bytelen] = Ptr{Void}[]
        p = Ptr{Void}[0]
        cudaMalloc(p, bytelen)
        ptr = p[1]
    end
    CudaPtr{T}(Ptr{T}(ptr), bytelen, dev)
end

free(p::CudaPtr) = push!(freeptrs[p.dev+1][p.bytelen], Ptr{Void}(p.ptr))

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
