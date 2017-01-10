type TensorDesc
    ptr::Ptr{Void}

    function TensorDesc{T,N}(x::CuArray{T,N})
        c_size = Cint[size(x,i) for i=ndims(x):-1:1]
        c_strides = Cint[stride(x,i) for i=ndims(x):-1:1]
        p = Ptr{Void}[0]
        cudnnCreateTensorDescriptor(p)
        cudnnSetTensorNdDescriptor(p[1], datatype(T), N, c_size, c_strides)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyTensorDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::TensorDesc) = desc.ptr

type ActivationDesc
    ptr::Ptr{Void}

    function ActivationDesc(mode; relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
        p = Ptr{Void}[0]
        cudnnCreateActivationDescriptor(p)
        cudnnSetActivationDescriptor(p[1], mode, relu_nanopt, relu_ceiling)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyActivationDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr

type ConvDesc
    ptr::Ptr{Void}

    function ConvDesc{T,N}(::Type{T}, pads::NTuple{N,Int}, strides; mode=CUDNN_CROSS_CORRELATION)
        p = Ptr{Void}[0]
        cudnnCreateConvolutionDescriptor(p)
        c_pads = Cint[pads[i] for i=N:-1:1]
        c_strides = Cint[strides[i] for i=N:-1:1]
        c_upscale = fill(Cint(1), N)
        cudnnSetConvolutionNdDescriptor(p[1], N, c_pads, c_strides, c_upscale, mode, datatype(T))
        desc = new(p[1])
        finalizer(desc, cudnnDestroyConvolutionDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ConvDesc) = desc.ptr

type DropoutDesc
    ptr::Ptr{Void}

    function DropoutDesc()
        p = Ptr{Void}[0]
        cudnnCreateDropoutDescriptor(p)
        desc = new(p[1])
        finalizer(desc, cudnnDestroyDropoutDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::DropoutDesc) = desc.ptr

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
