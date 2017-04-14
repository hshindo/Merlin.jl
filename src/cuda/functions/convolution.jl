function forward{T}(::typeof(conv), x::CuArray{T}, w::CuArray{T,4}, b::CuVector{T},
    pads::NTuple{2,Int}, strides::NTuple{2,Int})

    outdims = ntuple(length(pads)) do i
        (size(x.data,i) + 2pads[i] - size(w.data,i)) ÷ strides[i] + 1
    end
    y = similar(x.data, outdims..., size(w.data,N), size(x.data,N))
    desc = ConvDesc(T, pads, strides)
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

    cudnnConvolutionForward(handle(x), T[alpha], xdesc, x, wdesc, w, desc,
        algo, workspace, worksize, T[beta], ydesc, y)

    function backward!(gy, gx, gw, gb)
        isvoid(gw) || CUDNN.∇convolution_filter!(x.data, gy, desc, w.grad, beta=1.0)
        isvoid(gx) || CUDNN.∇convolution_data!(w.data, gy, desc, x.grad, beta=1.0)
        if !isvoid(gb)
            cudnnConvolutionBackwardBias(h, T[1], dydesc, dy, T[1], dbdesc, db)
        end
    end

    y, backward!
end

function ∇convolution_bias!(dy, db)
    dydesc = TensorDesc(dy)
    dbdesc = TensorDesc(db)
    T = eltype(dy)
    cudnnConvolutionBackwardBias(handle(dy), T[1], dydesc, dy, T[1], dbdesc, db)
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
