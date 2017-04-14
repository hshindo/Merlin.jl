export
    CUDNN_POOLING_MAX,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
    
    CUDNN_NOT_PROPAGATE_NAN,
    CUDNN_PROPAGATE_NAN

function pooling_desc(mode, window, padding, stride, maxpoolingNanOpt)
    N = length(window)
    p = Ptr{Void}[0]
    cudnnCreatePoolingDescriptor(p)
    cwindow = Cint[window[i] for i=N:-1:1]
    cpadding = Cint[padding[i] for i=N:-1:1]
    cstride = Cint[stride[i] for i=N:-1:1]
    cudnnSetPoolingNdDescriptor(p[1], mode, maxpoolingNanOpt, N, cwindow, cpadding, cstride)
    p[1]
end

function pooling(mode, window, padding, stride, x; maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN, alpha=1.0, beta=0.0)
    N = length(window)
    outdims = [(size(x,i) + 2padding[i] - window[i]) ÷ stride[i] + 1 for i=1:N]
    y = similar(x, outdims..., size(x,N+1), size(x,N+2))

    T = eltype(x)
    h = handle(x)
    poolingdesc = pooling_desc(mode, window, padding, stride, maxpoolingNanOpt)
    xdesc = tensor_desc(x)
    ydesc = tensor_desc(y)
    cudnnPoolingForward(h, poolingdesc, T[alpha], xdesc, x, T[beta], ydesc, y)

    cudnnDestroyPoolingDescriptor(poolingdesc)
    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    y
end

function ∇pooling!(mode, windims, padding, stride, y, dy, x, dx;
    maxpoolingNanOpt=CUDNN_NOT_PROPAGATE_NAN, alpha=1.0, beta=0.0)

    h = gethandle(device(x))
    poolingdesc = pooling_desc(mode, window, padding, stride, maxpoolingNanOpt)
    ydesc = tensor_desc(y)
    dydesc = tensor_desc(dy)
    xdesc = tensor_desc(x)
    dxdesc = tensor_desc(dx)
    cudnnPoolingBackward(h, poolingdesc, T[alpha], ydesc, y, dydesc, dy, xdesc, x,
    T[beta], dxdesc, dx)

    cudnnDestroyPoolingDescriptor(poolingdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(dxdesc)
end
