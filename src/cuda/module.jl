const CuModule_t = Ptr{Void}

mutable struct CuModule
    ptr::CuModule_t

    function CuModule(ptx::String)
        ref = Ref{CuModule_t}()
        @apicall :cuModuleLoadData (Ptr{CuModule_t},Ptr{Void}) ref pointer(ptx)
        mod = new(ref[])
        finalizer(mod, m -> @apicall :cuModuleUnload (CuModule_t,) m)
        mod
    end
end

Base.unsafe_convert(::Type{CuModule_t}, m::CuModule) = m.ptr

#=
function CuModule(image::Vector{UInt8})
    ref = Ref{Void}()
    cuModuleLoadData(ref, image)
    #cuModuleLoadDataEx(p, image, 0, CUjit_option[], Ptr{Void}[])
    CuModule(ref[])
end
=#
