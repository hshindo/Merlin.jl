mutable struct Handle
    ptr::Ptr{Cvoid}

    function Handle()
        ref = Ref{Ptr{Cvoid}}()
        @cudnn :cudnnCreate (Ptr{Ptr{Cvoid}},) ref
        h = new(ref[])
        finalizer(h) do x
            @cudnn :cudnnDestroy (Ptr{Cvoid},) x
        end
        h
    end
end

Base.cconvert(::Type{Ptr{Cvoid}}, h::Handle) = h.ptr
Base.isequal(x::Handle, y::Handle) = isequal(x.ptr, y.ptr)

const HANDLES = Dict{Int,Handle}()

function gethandle()
    dev = CUDA.getdevice()
    get!(HANDLES, dev) do
        Handle()
    end
end
