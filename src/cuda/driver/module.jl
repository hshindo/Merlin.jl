mutable struct CuModule
    ptr::Ptr{Cvoid}
    ctx::CuContext
end

function CuModule(ptx::String)
    ref = Ref{Ptr{Cvoid}}()
    @apicall :cuModuleLoadData (Ptr{Ptr{Cvoid}},Ptr{Cvoid}) ref pointer(ptx)
    mod = CuModule(ref[], getcontext())
    finalizer(unload, mod)
    mod
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, m::CuModule) = m.ptr

function unload(mod::CuModule)
    setcontext(mod.ctx) do
        @apicall :cuModuleUnload (Ptr{Cvoid},) mod
    end
end
