mutable struct CuModule
    ptr::Ptr{Void}
    ctx::CuContext
end

function CuModule(ptx::String)
    ref = Ref{Ptr{Void}}()
    @apicall :cuModuleLoadData (Ptr{Ptr{Void}},Ptr{Void}) ref pointer(ptx)
    mod = CuModule(ref[], getcontext())
    finalizer(mod, unload)
    mod
end

Base.unsafe_convert(::Type{Ptr{Void}}, m::CuModule) = m.ptr

function unload(mod::CuModule)
    setcontext(mod.ctx) do
        @apicall :cuModuleUnload (Ptr{Void},) mod
    end
end
