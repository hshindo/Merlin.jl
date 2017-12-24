const CuModule_t = Ptr{Void}

mutable struct CuModule
    ptr::CuModule_t

    function CuModule(ptr)
        m = new(ptr)
        finalizer(m) do x
            @apicall :cuModuleUnload (CuModule_t,) x
        end
        m
    end
end

Base.unsafe_convert(::Type{CuModule_t}, m::CuModule) = m.ptr

function CuModule(image::Vector{UInt8})
    ref = Ref{Void}()
    cuModuleLoadData(ref, image)
    #cuModuleLoadDataEx(p, image, 0, CUjit_option[], Ptr{Void}[])
    CuModule(ref[])
end
