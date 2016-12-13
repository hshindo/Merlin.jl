export
    dnnAlgorithmConvolutionGemm,
    dnnAlgorithmConvolutionDirect,
    dnnAlgorithmConvolutionFFT,
    dnnBorderZeros, dnnBorderExtrapolation # dnnBorder_t

function convolution{T}(x::Array{T}, w::Array{T};
    algo=dnnAlgorithmConvolutionDirect, border=dnnBorderZeros)

    p = Ptr{Void}[0]
    dnnPrimitiveAttributesCreate_F32(p)
    attr = p[1]

    y = zeros(T, 4, 3, 2, 2)
    p = Ptr{Void}[0]
    src_csize = Csize_t[size(x,i) for i=1:ndims(x)]
    dst_csize = Csize_t[size(y,i) for i=1:ndims(y)]
    filter_csize = Csize_t[size(w,i) for i=1:ndims(w)]
    cstrides = Csize_t[1, 1]
    offset = Cint[0, 0]
    dnnConvolutionCreateForward_F32(p, attr, algo, 4, src_csize, dst_csize, filter_csize,
    cstrides, offset, border)
    conv = p[1]

    resources = Ptr{Void}[pointer(x), pointer(y), pointer(w)]
    dnnExecute_F32(conv, resources)

    dnnPrimitiveAttributesDestroy_F32(attr)
    dnnDelete_F32(conv)
    y
end
