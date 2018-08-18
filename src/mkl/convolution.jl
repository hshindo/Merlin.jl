export
    dnnAlgorithmConvolutionGemm,
    dnnAlgorithmConvolutionDirect,
    dnnAlgorithmConvolutionFFT,
    dnnBorderZeros, dnnBorderExtrapolation # dnnBorder_t

function convolution_desc{N}(x::Array, w::Array,
    padding::NTuple{N,Int}, strides::NTuple{N,Int};
    algo=dnnAlgorithmConvolutionDirect, border=dnnBorderZeros)

    csize_x = Csize_t[size(x)...]
    dims = ntuple(i -> (size(x,i)+2padding[i]-size(w,i)) ÷ strides[i] + 1, N)
    csize_y = Csize_t[dims..., size(w,ndims(w)), size(x,ndims(x))]
    csize_w = Csize_t[size(w)...]
    cstrides = Csize_t[strides...]
    cpadding = Cint[padding...]
    (algo, ndims(x), csize_x, csize_y, csize_w, cstrides, cpadding, border)
end

function convolution{T}(x::Array{T}, w::Array{T}, padding, strides)
    desc = convolution_desc(x, w, padding, strides)
    y = Array{T}(desc[4]...)
    p = Ptr{Cvoid}[0]
    attr = dnnPrimitiveAttributesCreate()
    dnnConvolutionCreateForward_F32(p, attr, desc...)
    p_conv = p[1]

    resources = Ptr{Cvoid}[pointer(x),pointer(y),pointer(w)]
    dnnExecute(p_conv, resources)

    dnnPrimitiveAttributesDestroy(attr)
    dnnDelete(p_conv)
    y
end

function ∇convolution_data{T}(x::Array{T}, w::Array{T}, dy::Array{T}, padding, strides)
    p = Ptr{Cvoid}[0]
    attr = dnnPrimitiveAttributesCreate()
    desc = convolution_desc(x, w, padding, strides)
    dnnConvolutionCreateBackwardData_F32(p, attr, desc...)
    p_conv = p[1]

    dx = similar(x)
    resources = Ptr{Cvoid}[0,0,pointer(w),0,pointer(dx),0,0,pointer(dy)]
    dnnExecute(p_conv, resources)

    dnnPrimitiveAttributesDestroy(attr)
    dnnDelete(p_conv)
    dx
end

function ∇convolution_filter{T}(x::Array{T}, w::Array{T}, dy::Array{T}, padding, strides)
    p = Ptr{Cvoid}[0]
    attr = dnnPrimitiveAttributesCreate()
    desc = convolution_desc(x, w, padding, strides)
    dnnConvolutionCreateBackwardFilter_F32(p, attr, desc...)
    p_conv = p[1]

    dw = similar(w)
    resources = Ptr{Cvoid}[pointer(x),0,0,0,0,pointer(dw),0,pointer(dy)]
    dnnExecute(p_conv, resources)

    dnnPrimitiveAttributesDestroy(attr)
    dnnDelete(p_conv)
    dw
end

function ∇convolution_bias{T}(x::Array{T}, w::Array{T}, b::Array{T}, dy::Array{T}, padding, strides)
    p = Ptr{Cvoid}[0]
    attr = dnnPrimitiveAttributesCreate()
    algorithm,dimension,dstSize
    desc = convolution_desc(x, w, padding, strides)
    dnnConvolutionCreateBackwardBias_F32(p, attr, desc...)
    p_conv = p[1]

    db = similar(b)
    resources = Ptr{Cvoid}[0,0,0,0,0,0,pointer(db),pointer(dy)]
    dnnExecute(p_conv, resources)

    dnnPrimitiveAttributesDestroy(attr)
    dnnDelete(p_conv)
    db
end
