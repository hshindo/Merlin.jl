mutable struct CuContext
    ptr::Ptr{Cvoid}

    function CuContext(dev::Int)
        ref = Ref{Ptr{Cvoid}}()
        @apicall :cuCtxCreate (Ptr{Ptr{Cvoid}},Cuint,Cint) ref 0 dev
        ctx = new(ref[])
        finalizer(destroy, ctx)
        ctx
    end
end

const CONTEXTS = Dict{Int,CuContext}()

Base.:(==)(a::CuContext, b::CuContext) = a.ptr == b.ptr
Base.hash(ctx::CuContext, h::UInt) = hash(ctx.ptr, h)
Base.unsafe_convert(::Type{Ptr{Cvoid}}, ctx::CuContext) = ctx.ptr

function destroy(ctx::CuContext)
    setcontext(ctx)
    @apicall :cuCtxDestroy (Ptr{Cvoid},) ctx
end

function getcontext()
    CONTEXTS[getdevice()]
    #ref = Ref{Ptr{Cvoid}}()
    #@apicall :cuCtxGetCurrent (Ptr{Ptr{Cvoid}},) ref
    #CuContext(ref[])
end

function setcontext(ctx::CuContext)
    @apicall :cuCtxSetCurrent (Ptr{Cvoid},) ctx.ptr
end

function synchronize()
    @apicall :cuCtxSynchronize ()
    empty!(ALLOCATED)
end
