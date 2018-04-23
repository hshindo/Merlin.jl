mutable struct Handle
    ptr::Ptr{Void}

    function Handle()
        ref = Ref{Ptr{Void}}()
        @cudnn :cudnnCreate (Ptr{Ptr{Void}},) ref
        h = new(ref[])
        finalizer(h, x -> @cudnn :cudnnDestroy (Ptr{Void},) x)
        h
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, h::Handle) = h.ptr

const HANDLES = Array{Handle}(ndevices())

function gethandle()
    dev = getdevice()
    isassigned(HANDLES,dev) || (HANDLES[dev+1] = Handle())
    HANDLES[dev+1]
end
