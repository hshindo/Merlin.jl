type CuModule
    ptr::Ptr{Void}

    function CuModule(ptr)
        m = new(ptr)
        finalizer(m, cuModuleUnload)
        m
    end
end

function CuModule(image::Vector{UInt8})
    p = Ptr{Void}[0]
    cuModuleLoadData(p, image)
    #cuModuleLoadDataEx(p, image, 0, CUjit_option[], Ptr{Void}[])
    CuModule(p[1])
end

Base.unsafe_convert(::Type{Ptr{Void}}, m::CuModule) = m.ptr
