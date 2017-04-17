function rnn_training!{T}(xdims, x::CuArray{T}, hx::CuArray{T}, cx::CuArray{T},
    droprate, input_t, dir_t, net_t; seed=0)

    xdesc = tensor_desc(CuArray(T, xdims[1]...))
    xdescs = fill(xdesc, length(xdims))
    for i=1:length(xdims) xdescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    hxdesc = tensor_desc(hx)
    cxdesc = tensor_desc(cx)

    y = similar(x)
    hy = similar(hx)
    cy = similar(cx)
    ydescs = similar(xdescs)
    for i=1:length(xdims) ydescs[i] = tensor_desc(CuArray(T, xdims[i])) end
    hydesc = tensor_desc(hy)
    cydesc = tensor_desc(cy)

    h = gethandle(device(x))
    rnndesc, dropdesc, dropstate = rnn_desc(x, size(hx,2), size(hx,4), input_t,
        dir_t, net_t, droprate, seed)
    wsize_p = Cint[0]
    cudnnGetRNNParamsSize(h, rnndesc, xdesc, wsize_p, datatype(T))
    wsize = wsize_p[1]
    w = curand(T, 1, 1, 1, Int(wsize/(T.size)))
    wdesc = filter_desc(w)

    worksize_p = Cint[0]
    cudnnGetRNNWorkspaceSize(h, rnndesc, Cint(length(xdescs)), xdescs, worksize_p)
    worksize = worksize_p[1]
    workspace = CuArray(Int8, Int(worksize))

    trainsize_p = Cint[0]
    cudnnGetRNNTrainingReserveSize(h, rnndesc, Cint(length(xdescs)), xdescs, trainsize_p)
    trainsize = trainsize_p[1]
    trainspace = CuArray(Int8, Int(trainsize))

    mdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(mdesc_p)
    mdesc = mdesc_p[1]
    m_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerMatrixParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), mdesc, m_p)
    m = m_p[1]

    bdesc_p = Ptr{Void}[0]
    cudnnCreateFilterDescriptor(bdesc_p)
    bdesc = bdesc_p[1]
    b_p = Ptr{Void}[0]
    cudnnGetRNNLinLayerBiasParams(h, rnndesc, Cint(0), xdesc, wdesc, w,
        Cint(0), bdesc, b_p)
    b = b_p[1]

    cudnnRNNForwardTraining(h, rnndesc, Cint(length(xdescs)), xdescs, x, hxdesc,
        hx, cxdesc, cx, wdesc, w, ydescs, y, hydesc, hy, cydesc, cy, workspace,
        worksize, trainspace, trainsize)

    cudnnDestroyFilterDescriptor(bdesc)
    cudnnDestroyFilterDescriptor(mdesc)
    cudnnDestroyFilterDescriptor(wdesc)
    cudnnDestroyRNNDescriptor(rnndesc)
    cudnnDestroyTensorDescriptor(cydesc)
    cudnnDestroyTensorDescriptor(hydesc)
    for desc in ydescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(cxdesc)
    cudnnDestroyTensorDescriptor(hxdesc)
    for desc in xdescs cudnnDestroyTensorDescriptor(desc) end
    cudnnDestroyTensorDescriptor(xdesc)
    w, y, hy, cy, dropdesc, dropstate
end
