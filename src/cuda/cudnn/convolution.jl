export CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION

function convolution{T}(x::CuArray{T}, w::CuArray{T}, desc::ConvDesc; alpha=1.0, beta=0.0)
    N = length(desc.padding)
    outdims = ntuple(N) do i
        (size(x,i) + 2*desc.padding[i] - size(w,i)) ÷ desc.strides[i] + 1
    end
    y = similar(x, outdims..., size(w,N+2), size(x,N+2))

    h = handle(x)
    xdesc = TensorDesc(x)
    wdesc = TensorDesc(w)
    ydesc = TensorDesc(y)

    p = cudnnConvolutionFwdAlgo_t[0]
    cudnnGetConvolutionForwardAlgorithm(h, xdesc, wdesc, desc, ydesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, p)
    algo = p[1]

    p = Cint[0]
    cudnnGetConvolutionForwardWorkspaceSize(h, xdesc, wdesc, desc, ydesc, algo, p)
    worksize = p[1]
    workspace = CuArray{Int8}(Int(worksize))

    cudnnConvolutionForward(h, T[alpha], xdesc, x, wdesc, w, desc,
        algo, workspace, worksize, T[beta], ydesc, y)
    y
end

function ∇convolution_bias!(dy, db; alpha=1.0, beta=0.0)
    T = eltype(dy)
    h = handle(dy)
    dydesc = TensorDesc(dy)
    dbdesc = TensorDesc(db)
    cudnnConvolutionBackwardBias(h, T[alpha], dydesc, dy, T[beta], dbdesc, db)
end

function ∇convolution_filter!(x, dy, desc::ConvDesc, dw; alpha=1.0, beta=0.0)
    T = eltype(dy)
    h = handle(dy)
    p = cudnnConvolutionBwdFilterAlgo_t[0]
    cudnnGetConvolutionBackwardFilterAlgorithm(h, xdesc, dydesc, desc, dwdesc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, p)
    algo = p[1]

    p = Cint[0]
    cudnnGetConvolutionBackwardFilterWorkspaceSize(h, xdesc, dydesc, desc, dwdesc, algo, p)
    worksize = p[1]
    workspace = CuArray{Int8}(Int(p[1]))

    cudnnConvolutionBackwardFilter(h, T[alpha], xdesc, x, dydesc, dy, desc,
        algo, workspace, worksize, T[beta], dwdesc, dw)
end

function ∇convolution_data!(w, dy, desc::ConvDesc, dx; alpha=1.0, beta=0.0)
    T = eltype(dy)
    h = handle(dy)
    p = cudnnConvolutionBwdDataAlgo_t[0]
    cudnnGetConvolutionBackwardDataAlgorithm(h, wdesc, dydesc, desc, dxdesc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, p)
    algo = p[1]

    p = Cint[0]
    cudnnGetConvolutionBackwardDataWorkspaceSize(h, wdesc, dydesc, desc,
        dxdesc, algo, p)
    worksize = p[1]
    workspace = CuArray{Int8}(Int(worksize))

    cudnnConvolutionBackwardData(h, T[alpha], wdesc, w, dydesc, dy, desc,
        algo, workspace, worksize, T[beta], dxdesc, dx)
end
