export relu

function relu{T,N}(x::Array{T,N})
    p = Ptr{Void}[0]
    dnnPrimitiveAttributesCreate_F32(p)
    attributes = p[1]

    p = Ptr{Void}[0]
    csize = [size(x,i) for i=1:ndims(x)]
    cstrides = [stride(x,i) for i=1:ndims(x)]
    dnnLayoutCreate_F32(p, ndims(x), csize, cstrides)
    layout = p[1]

    p = Ptr{Void}[0]
    dnnReLUCreateForward_F32(p, attributes, layout, Cfloat(0))
    primitive = p[1]

    y = zeros(x)
    dnnConversionExecute_F32(primitive, x, y)

    dnnPrimitiveAttributesDestroy_F32(attributes)
    dnnLayoutDelete_F32(layout)
    dnnDelete_F32(primitive)
    y
end
