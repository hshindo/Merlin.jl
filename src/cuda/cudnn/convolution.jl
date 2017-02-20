export CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION

type ConvDesc
    ptr::Ptr{Void}

    function ConvDesc{T,N}(::Type{T}, pads::NTuple{N,Int}, strides; mode=CUDNN_CROSS_CORRELATION)
        p = Ptr{Void}[0]
        cudnnCreateConvolutionDescriptor(p)
        c_pads = Cint[pads[i] for i=N:-1:1]
        c_strides = Cint[strides[i] for i=N:-1:1]
        c_upscale = fill(Cint(1), N)
        cudnnSetConvolutionNdDescriptor(p[1], N, c_pads, c_strides, c_upscale, mode, datatype(T))
        desc = new(p[1])
        finalizer(desc, cudnnDestroyConvolutionDescriptor)
        desc
    end
end

Base.unsafe_convert(::Type{Ptr{Void}}, desc::ConvDesc) = desc.ptr

function convolution!(x, w, desc, y; alpha=1.0, beta=0.0)
    xdesc = TensorDesc(x)
    wdesc = TensorDesc(w)
    ydesc = TensorDesc(y)

    p = cudnnConvolutionFwdAlgo_t[0]
    cudnnGetConvolutionForwardAlgorithm(handle(x), xdesc, wdesc, desc, ydesc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, p)
    algo = p[1]

    p = Cint[0]
    cudnnGetConvolutionForwardWorkspaceSize(handle(x), xdesc, wdesc, desc, ydesc, algo, p)
    worksize = p[1]
    workspace = CuArray{Int8}(Int(worksize))

    T = eltype(x)
    cudnnConvolutionForward(handle(x), T[alpha], xdesc, x, wdesc, w, desc,
        algo, workspace, worksize, T[beta], ydesc, y)
    y
end

function ∇convolution_bias!(dy, db; alpha=1.0, beta=0.0)
    dydesc = TensorDesc(dy)
    dbdesc = TensorDesc(db)
    T = eltype(dy)
    cudnnConvolutionBackwardBias(handle(dy), T[alpha], dydesc, dy, T[beta], dbdesc, db)
end

function ∇convolution_filter!(x, dy, desc::ConvDesc, dw; alpha=1.0, beta=0.0)
    h = handle(x)
    p = cudnnConvolutionBwdFilterAlgo_t[0]
    cudnnGetConvolutionBackwardFilterAlgorithm(h, xdesc, dydesc, desc, dwdesc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, p)
    algo = p[1]

    p = Cint[0]
    cudnnGetConvolutionBackwardFilterWorkspaceSize(h, xdesc, dydesc, desc, dwdesc, algo, p)
    worksize = p[1]
    workspace = CuArray{Int8}(Int(p[1]))

    T = eltype(dy)
    cudnnConvolutionBackwardFilter(h, T[alpha], xdesc, x, dydesc, dy, desc,
        algo, workspace, worksize, T[beta], dwdesc, dw)
end

function ∇convolution_data!(w, dy, desc::ConvDesc, dx; alpha=1.0, beta=0.0)
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

    T = eltype(dy)
    cudnnConvolutionBackwardData(h, T[alpha], wdesc, w, dydesc, dy, desc,
        algo, workspace, worksize, T[beta], dxdesc, dx)
end
