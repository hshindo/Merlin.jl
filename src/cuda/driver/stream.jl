export CuStream

const CU_STREAM_DEFAULT = Cint(0)
const CU_STREAM_NON_BLOCKING = Cint(1)

mutable struct CuStream
    ptr::Ptr{Cvoid}

    function CuStream(flags::Cint=CU_STREAM_DEFAULT)
        ref = Ref{Ptr{Cvoid}}()
        @apicall :cuStreamCreate (Ptr{Ptr{Cvoid}},Cuint) ref flags
        s = new(ref[], getcontext())
        finalizer(s, destroy)
        s
    end
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, s::CuStream) = s.ptr

function destroy(s::CuStream)
    setcontext(s.ctx) do
        @apicall :cuStreamDestroy (Ptr{Cvoid},) s
    end
end

synchronize(s::CuStream) = @apicall :cuStreamSynchronize (Ptr{Cvoid},) s
