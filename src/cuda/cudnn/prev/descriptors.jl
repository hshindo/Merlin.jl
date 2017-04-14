





type FilterDesc
    ptr::Ptr{Void}

    function FilterDesc()
        csize = Cint[size(a,i) for i=ndims(a):-1:1]
        p = Ptr{Void}[0]
        cudnnCreateFilterDescriptor(p)
        cudnnSetFilterNdDescriptor(p[1], datatype(T), format, ndims(a), csize)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyFilterDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::FilterDesc) = desc.ptr
