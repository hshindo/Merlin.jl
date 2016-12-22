reshape4d{T}(x::CuArray{T,1}) = reshape(x, size(x,1), 1, 1, 1)
reshape4d{T}(x::CuArray{T,2}) = reshape(x, size(x,1), size(x,2), 1, 1)
reshape4d{T}(x::CuArray{T,3}) = reshape(x, size(x,1), size(x,2), size(x,3), 1)
reshape4d{T}(x::CuArray{T,4}) = x

reshape4d_r{T}(x::CuArray{T,1}) = reshape(x, 1, 1, 1, size(x,1))
reshape4d_r{T}(x::CuArray{T,2}) = reshape(x, 1, 1, size(x,1), size(x,2))
reshape4d_r{T}(x::CuArray{T,3}) = reshape(x, 1, size(x,1), size(x,2), size(x,3))
reshape4d_r{T}(x::CuArray{T,4}) = x

type TensorDesc
    ptr::Ptr{Void}

    function TensorDesc(x::CuArray)
        csize = Cint[size(a,i) for i=ndims(a):-1:1]
        cstrides = Cint[stride(a,i) for i=ndims(a):-1:1]
        p = Ptr{Void}[0]
        cudnnCreateTensorDescriptor(p)
        cudnnSetTensorNdDescriptor(p[1], datatype(T), ndims(a), csize, cstrides)
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

type ConvDesc{N}
    ptr::Ptr{Void}
    padding::NTuple{N,Int}
    strides::NTuple{N,Int}

    function ConvDesc(T::Type, padding, strides; mode=CUDNN_CROSS_CORRELATION)
        N = length(padding)
        p = Ptr{Void}[0]
        cudnnCreateConvolutionDescriptor(p)
        cpadding = Cint[padding[i] for i=N:-1:1]
        cstrides = Cint[stride[i] for i=N:-1:1]
        cupscale = fill(Cint(1), N)
        cudnnSetConvolutionNdDescriptor(p[1], N, cpadding, cstrides, cupscale, mode, datatype(T))
        desc = new(p[1], padding, strides)
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
