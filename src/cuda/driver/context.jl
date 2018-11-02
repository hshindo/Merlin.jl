mutable struct CuContext
    ptr::Ptr{Cvoid}
end

function CuContext(dev::Int)
    ref = Ref{Ptr{Cvoid}}()
    @apicall :cuCtxCreate (Ptr{Ptr{Cvoid}},Cuint,Cint) ref 0 dev
    ctx = CuContext(ref[])
    # finalizer(ctx, destroy)
    ctx
end

Base.:(==)(a::CuContext, b::CuContext) = a.ptr == b.ptr
Base.hash(ctx::CuContext, h::UInt) = hash(ctx.ptr, h)
Base.unsafe_convert(::Type{Ptr{Cvoid}}, ctx::CuContext) = ctx.ptr

function destroy(ctx::CuContext)
    setcontext(ctx)
    @apicall :cuCtxDestroy (Ptr{Cvoid},) ctx
end

function getcontext()
    ref = Ref{Ptr{Cvoid}}()
    @apicall :cuCtxGetCurrent (Ptr{Ptr{Cvoid}},) ref
    CuContext(ref[])
end

function setcontext(ctx::CuContext)
    @apicall :cuCtxSetCurrent (Ptr{Cvoid},) ctx
end
function setcontext(f::Function, ctx::CuContext)
    oldctx = getcontext()
    setcontext(ctx)
    f()
    setcontext(oldctx)
end

function synchronize()
    @apicall :cuCtxSynchronize ()
    empty!(ALLOCATED)
    empty!(CUDNN.ALLOCATED)
end
