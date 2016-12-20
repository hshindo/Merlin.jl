export
    CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL, # mode
    CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG # algorithm

function softmax(algo, mode, x; alpha=1.0, beta=0.0)
    T = eltype(x)
    h = handle(x)
    y = similar(x)
    xdesc = tensor_desc(reshape4d_r(x))
    ydesc = tensor_desc(reshape4d_r(y))
    cudnnSoftmaxForward(h, algo, mode, T[alpha], xdesc, x, T[beta], ydesc, y)

    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    y
end

function ∇softmax!(algo, mode, y, dy, dx; alpha=1.0, beta=0.0)
    T = eltype(y)
    h = handle(y)
    ydesc = tensor_desc(reshape4d_r(y))
    dydesc = tensor_desc(reshape4d_r(dy))
    dxdesc = tensor_desc(reshape4d_r(dx))
    cudnnSoftmaxBackward(h, algo, mode, T[alpha], ydesc, y, dydesc, dy,
        T[beta], dxdesc, dx)

    cudnnDestroyTensorDescriptor(ydesc)
    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyTensorDescriptor(dxdesc)
    dx
end
