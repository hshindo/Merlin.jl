export
    CUDNN_ACTIVATION_SIGMOID,
    CUDNN_ACTIVATION_RELU,
    CUDNN_ACTIVATION_TANH,
    CUDNN_ACTIVATION_CLIPPED_RELU

"""
reluNanOpt: whether propagates NaN or not
reluCeiling: floating point number to specify the clipping threshold
when the activation mode is set to CUDNN_ACTIVATION_CLIPPED_RELU.
"""
function activation{T}(desc, x::CuArray{T}; alpha=1.0, beta=0.0)
    h = handle(x)
    y = similar(x)
    xdesc = TensorDesc(redim(x,4))
    ydesc = TensorDesc(redim(y,4))
    cudnnActivationForward(h, desc, T[alpha], xdesc, x, T[beta], ydesc, y)
    y
end

function ∇activation!(desc, y, dy, x, dx; alpha=1.0, beta=0.0)
    T = eltype(y)
    h = handle(x)
    ydesc = TensorDesc(redim(y,4))
    dydesc = TensorDesc(redim(dy,4))
    xdesc = TensorDesc(redim(x,4))
    dxdesc = TensorDesc(redim(dx,4))
    cudnnActivationBackward(h, desc, T[alpha], ydesc, y, dydesc, dy,
        xdesc, x, T[beta], dxdesc, dx)
end
