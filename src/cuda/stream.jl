const CuStream_t = Ptr{Void}

mutable struct CuStream
    ptr::CuStream_t

    function CuStream(flags::Int=0)
        ref = Ref{CuStream_t}()
        @apicall :cuStreamCreate (Ptr{CuStream_t},Cuint) ref flags
        s = CuStream(ref[])
        finalizer(s) do x
            @apicall :cuStreamDestroy (CuStream_t,) x
        end
        s
    end
end

Base.unsafe_convert(::Type{CuStream_t}, s::CuStream) = s.ptr
Base.:(==)(a::CuStream, b::CuStream) = a.ptr == b.ptr
Base.hash(s::CuStream, h::UInt) = hash(s.ptr, h)

const CuDefaultStream() = CuStream(0)

synchronize(s::CuStream) = @apicall :cuStreamSynchronize (CuStream_t,) s
