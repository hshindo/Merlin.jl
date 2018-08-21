mutable struct Handle
    ptr::Ptr{Cvoid}

    function Handle()
        ref = Ref{Ptr{Cvoid}}()
        @cublas :cublasCreate (Ptr{Ptr{Cvoid}},) ref
        h = new(ref[])
        finalizer(h, x -> @cublas :cublasDestroy (Ptr{Cvoid},) x)
        h
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, h::Handle) = h.ptr

const HANDLES = Array{Handle}(undef, ndevices())

function gethandle()
    dev = getdevice()
    isassigned(HANDLES,dev) || (HANDLES[dev+1] = Handle())
    HANDLES[dev+1]
end
