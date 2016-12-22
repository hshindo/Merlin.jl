export
    CUDNN_ACTIVATION_SIGMOID,
    CUDNN_ACTIVATION_RELU,
    CUDNN_ACTIVATION_TANH,
    CUDNN_ACTIVATION_CLIPPED_RELU

"""
    activation

reluNanOpt: whether propagates NaN or not
reluCeiling: floating point number to specify the clipping threshold
when the activation mode is set to CUDNN_ACTIVATION_CLIPPED_RELU.
"""
function activation(desc, x; alpha=1.0, beta=0.0)
    T = eltype(x)
    h = handle(x)
    y = similar(x)
    xdesc = TensorDesc(reshape4d(x))
    ydesc = TensorDesc(reshape4d(y))
    cudnnActivationForward(h, desc, T[alpha], xdesc, x, T[beta], ydesc, y)
    y
end

function ∇activation!(desc, y, dy, x, dx; alpha=1.0, beta=0.0)
    T = eltype(y)
    h = handle(x)
    ydesc = TensorDesc(reshape4d(y))
    dydesc = TensorDesc(reshape4d(dy))
    xdesc = TensorDesc(reshape4d(x))
    dxdesc = TensorDesc(reshape4d(dx))
    cudnnActivationBackward(h, desc, T[alpha], ydesc, y, dydesc, dy,
        xdesc, x, T[beta], dxdesc, dx)
end
