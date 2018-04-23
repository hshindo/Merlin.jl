mutable struct CuContext
    ptr::Ptr{Void}
end

function CuContext(dev::Int)
    ref = Ref{Ptr{Void}}()
    @apicall :cuCtxCreate (Ptr{Ptr{Void}},Cuint,Cint) ref 0 dev
    ctx = CuContext(ref[])
    # finalizer(ctx, destroy)
    ctx
end

if AVAILABLE
    const CONTEXTS = [CuContext(i) for i=0:ndevices()-1]
end

Base.:(==)(a::CuContext, b::CuContext) = a.ptr == b.ptr
Base.hash(ctx::CuContext, h::UInt) = hash(ctx.ptr, h)
Base.unsafe_convert(::Type{Ptr{Void}}, ctx::CuContext) = ctx.ptr

function destroy(ctx::CuContext)
    setcontext(ctx)
    @apicall :cuCtxDestroy (Ptr{Void},) ctx
end

function getcontext()
    ref = Ref{Ptr{Void}}()
    @apicall :cuCtxGetCurrent (Ptr{Ptr{Void}},) ref
    CuContext(ref[])
end

function setcontext(ctx::CuContext)
    @apicall :cuCtxSetCurrent (Ptr{Void},) ctx
end
function setcontext(f::Function, ctx::CuContext)
    oldctx = getcontext()
    setcontext(ctx)
    f()
    setcontext(oldctx)
end

synchronize() = @apicall :cuCtxSynchronize ()
