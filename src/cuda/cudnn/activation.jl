struct ActivationDesc
    ptr::Ptr{Void}

    function ActivationDesc()
        p = Ptr{Void}[0]
        cudnnCreateActivationDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyActivationDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr
