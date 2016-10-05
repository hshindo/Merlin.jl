export
    CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_RELU,
    CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_CLIPPED_RELU

function activation_desc(mode, relu_nanopt, relu_ceiling)
    p = Ptr{Void}[0]
    cudnnCreateActivationDescriptor(p)
    cudnnSetActivationDescriptor(p[1], mode, relu_nanopt, relu_ceiling)
    p[1]
end

"""
reluNanOpt: whether propagates NaN or not
reluCeiling: floating point number to specify the clipping threshold
when the activation mode is set to CUDNN_ACTIVATION_CLIPPED_RELU.
"""
function activation(mode, x; relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
    T = eltype(x)
    h = handle(x)
    y = similar(x)
    adesc = activation_desc(mode, relu_nanopt, relu_ceiling)
    xdesc = tensor_desc(x)
    ydesc = tensor_desc(y)
    cudnnActivationForward(h, adesc, T[1], xdesc, x, T[0], ydesc, y)

    cudnnDestroyActivationDescriptor(adesc)
    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    y
end

function ∇activation!(mode, y, dy, x, dx; relu_nanopt=CUDNN_NOT_PROPAGATE_NAN, relu_ceiling=1.0)
    T = eltype(y)
    h = handle(x)
    adesc = activation_desc(mode, relu_nanopt, relu_ceiling)
    ydesc = tensor_desc(y)
    dydesc = tensor_desc(y)
    xdesc = tensor_desc(x)
    dxdesc = tensor_desc(dx)
    cudnnActivationBackward(h, adesc, T[1], ydesc, y, dydesc, dy,
    xdesc, x, T[0], dxdesc, dx)

    cudnnDestroyActivationDescriptor(adesc)
    cudnnDestroyTensorDescriptor(ydesc)
    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(dxdesc)
    dx
end
