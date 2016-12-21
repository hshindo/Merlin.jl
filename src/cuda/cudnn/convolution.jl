export CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION

type ConvDesc
    ptr::Ptr{Void}

    function ConvDesc(T::Type, padding, strides; mode=CUDNN_CROSS_CORRELATION)
        N = length(padding)
        p = Ptr{Void}[0]
        cudnnCreateConvolutionDescriptor(p)
        cpadding = Cint[padding[i] for i=N:-1:1]
        cstrides = Cint[stride[i] for i=N:-1:1]
        cupscale = fill(Cint(1), N)
        cudnnSetConvolutionNdDescriptor(p[1], N, cpadding, cstrides, cupscale, mode, datatype(T))
        desc = new(p[1])
        finalizer(desc, cudnnDestroyConvolutionDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ConvDesc) = desc.ptr

function convolution{T}(x::CuArray{T}, w::CuArray{T}, padding, strides;
    mode=CUDNN_CROSS_CORRELATION, alpha=1.0, beta=0.0)

    N = length(padding)
    outdims = ntuple(i -> (size(x,i)+2padding[i]-size(w,i)) ÷ stride[i] + 1, N)
    y = similar(x, outdims..., size(w,N+2), size(x,N+2))

    h = handle(x)
    xdesc = TensorDesc(x)
    wdesc = TensorDesc(w)
    convdesc = ConvDesc(T, padding, strides, mode)
    ydesc = TensorDesc(y)

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
    xdesc, wdesc, convdesc, ydesc, y
end

function ∇convolution_bias!(dy, db; alpha=1.0, beta=0.0)
    T = eltype(dy)
    h = handle(dy)
    dydesc = TensorDesc(dy)
    dbdesc = TensorDesc(db)
    cudnnConvolutionBackwardBias(h, T[alpha], dydesc, dy, T[beta], dbdesc, db)
    db
end

function ∇convolution_filter!(x, dy, convdesc, dw; alpha=1.0, beta=0.0)
    T = eltype(dy)
    h = handle(dy)
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
    dw
end

function ∇convolution_data!(w, dy, dx; alpha=1.0, beta=0.0)
    T = eltype(dy)
    h = handle(dy)
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
    dx
end
