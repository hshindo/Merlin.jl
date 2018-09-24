export relu

function relu(x::Array)
    p = Ptr{Cvoid}[0]
    dnnPrimitiveAttributesCreate_F32(p)
    attr = p[1]

    layout = create_layout(x)

    p = Ptr{Cvoid}[0]
    dnnReLUCreateForward_F32(p, attr, layout, Cfloat(0))
    primitive = p[1]

    y = similar(x)
    dnnConversionExecute_F32(primitive, x, y)

    dnnPrimitiveAttributesDestroy_F32(attr)
    dnnLayoutDelete_F32(layout)
    dnnDelete_F32(primitive)
    y
end
