export
    # cudnnActivationMode_t
    CUDNN_ACTIVATION_SIGMOID,
    CUDNN_ACTIVATION_RELU,
    CUDNN_ACTIVATION_TANH,
    CUDNN_ACTIVATION_CLIPPED_RELU

type ActivationDesc
    ptr::Ptr{Void}
end

function ActivationDesc()
    p = Ptr{Void}[0]
    cudnnCreateActivationDescriptor(p)
    desc = ActivationDesc(p[1])
    finalizer(desc, cudnnDestroyActivationDescriptor)
    desc
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ActivationDesc) = desc.ptr

function activation(mode, x; relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
    desc = ActivationDesc()
    cudnnSetActivationDescriptor(desc, mode, relu_nanopt, relu_ceiling)

    h = handle(x)
    T = eltype(x)
    xdesc = TensorDesc(x)
    y = similar(x)
    cudnnActivationForward(h, desc, T[1], xdesc, x, T[0], xdesc, y)

    function backward!(gy, gx)
        isvoid(gx) && return
        cudnnActivationBackward(h, desc, T[1], xdesc, y, xdesc, gy, xdesc, x, T[1], xdesc, gx)
    end
    y, backward!
end
