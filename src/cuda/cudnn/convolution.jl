export CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION

function convolution_desc(T::Type, padding, stride, mode)
    N = length(padding)
    p = Ptr{Void}[0]
    cudnnCreateConvolutionDescriptor(p)
    cpadding = Cint[padding[i] for i=N:-1:1]
    cstrides = Cint[stride[i] for i=N:-1:1]
    cupscale = fill(Cint(1), N)
    cudnnSetConvolutionNdDescriptor(p[1], N, cpadding, cstrides, cupscale, mode, datatype(T))
    p[1]
end

function convolution{T}(x::CuArray{T}, w::CuArray{T}, padding, stride;
    mode=CUDNN_CROSS_CORRELATION, alpha=1.0, beta=0.0)

    N = length(padding)
    outdims = ntuple(i -> (size(x,i)+2padding[i]-size(w,i)) ÷ stride[i] + 1, N)
    y = similar(x, outdims..., size(w,N+2), size(x,N+2))

    h = handle(x)
    xdesc = tensor_desc(x)
    wdesc = filter_desc(w)
    convdesc = convolution_desc(T, padding, stride, mode)
    ydesc = tensor_desc(y)

    algo_p = cudnnConvolutionFwdAlgo_t[0]
    cudnnGetConvolutionForwardAlgorithm(h, xdesc, wdesc, convdesc, ydesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, algo_p)
    algo = algo_p[1]

    worksize_p = Cint[0]
    cudnnGetConvolutionForwardWorkspaceSize(h, xdesc, wdesc, convdesc, ydesc, algo, worksize_p)
    worksize = worksize_p[1]
    workspace = CuArray{Int8}(Int(worksize))

    cudnnConvolutionForward(h, T[alpha], xdesc, x, wdesc, w, convdesc,
        algo, workspace, worksize, T[beta], ydesc, y)

    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyFilterDescriptor(wdesc)
    cudnnDestroyConvolutionDescriptor(convdesc)
    cudnnDestroyTensorDescriptor(ydesc)
    y
end

function ∇convolution_bias!(dy, db; mode=CUDNN_CROSS_CORRELATION, alpha=1.0, beta=0.0)
    T = eltype(dy)
    h = handle(dy)
    dydesc = tensor_desc(dy)
    dbdesc = tensor_desc(db)
    cudnnConvolutionBackwardBias(h, T[alpha], dydesc, dy, T[beta], dbdesc, db)

    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyTensorDescriptor(dbdesc)
    db
end

function ∇convolution_filter!(x, padding, stride, dy, dw; mode=CUDNN_CROSS_CORRELATION, alpha=1.0, beta=0.0)
    T = eltype(dy)
    h = handle(dy)
    xdesc = tensor_desc(x)
    dydesc = tensor_desc(dy)
    convdesc = convolution_desc(T, padding, stride, mode)
    dwdesc = filter_desc(dw)

    algo_p = cudnnConvolutionBwdFilterAlgo_t[0]
    cudnnGetConvolutionBackwardFilterAlgorithm(h, xdesc, dydesc, convdesc, dwdesc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, algo_p)
    algo = algo_p[1]

    worksize_p = Cint[0]
    cudnnGetConvolutionBackwardFilterWorkspaceSize(h, xdesc, dydesc, convdesc, dwdesc, algo, worksize_p)
    worksize = worksize_p[1]
    workspace = CuArray{Int8}(Int(worksize))

    cudnnConvolutionBackwardFilter(h, T[alpha], xdesc, x, dydesc, dy, convdesc,
        algo, workspace, worksize, T[beta], dwdesc, dw)

    cudnnDestroyTensorDescriptor(xdesc)
    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyConvolutionDescriptor(convdesc)
    cudnnDestroyFilterDescriptor(dwdesc)
    dw
end

function ∇convolution_data!(w, padding, stride, dy, dx; mode=CUDNN_CROSS_CORRELATION, alpha=1.0, beta=0.0)
    T = eltype(dy)
    h = handle(dy)
    wdesc = filter_desc(w)
    dydesc = tensor_desc(dy)
    convdesc = convolution_desc(T, padding, stride, mode)
    dxdesc = tensor_desc(dx)

    algo_p = cudnnConvolutionBwdDataAlgo_t[0]
    cudnnGetConvolutionBackwardDataAlgorithm(h, wdesc, dydesc, convdesc, dxdesc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, algo_p)
    algo = algo_p[1]

    worksize_p = Cint[0]
    cudnnGetConvolutionBackwardDataWorkspaceSize(h, wdesc, dydesc, convdesc,
        dxdesc, algo, worksize_p)
    worksize = worksize_p[1]
    workspace = CuArray{Int8}(Int(worksize))

    cudnnConvolutionBackwardData(h, T[alpha], wdesc, w, dydesc, dy, convdesc,
        algo, workspace, worksize, T[beta], dxdesc, dx)

    cudnnDestroyFilterDescriptor(wdesc)
    cudnnDestroyTensorDescriptor(dydesc)
    cudnnDestroyConvolutionDescriptor(convdesc)
    cudnnDestroyTensorDescriptor(dxdesc)
    dx
end
