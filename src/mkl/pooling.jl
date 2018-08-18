export
    dnnAlgorithmPoolingMax,
    dnnAlgorithmPoolingMin,
    dnnAlgorithmPoolingAvg

function pooling{T,N}(x::Array{T}, op, ksize::Tuple, padding::NTuple{N,Int}, strides::Tuple;
    border=dnnBorderZeros)
    attr = dnnPrimitiveAttributesCreate()
    layout = dnnLayoutCreate(x)
    cksize = Csize_t[ksize...]
    cstrides = Csize_t[strides...]
    coffsets = Cint[padding...]

    p = Ptr{Cvoid}[0]
    dnnPoolingCreateForward_F32(p, attr, op, layout, cksize, cstrides, coffsets, border)
    p_pool = p[1]

    p = Ptr{Cvoid}[0]
    dnnLayoutCreateFromPrimitive_F32(p, p_pool, dnnResourceWorkspace)
    p_workspace = p[1]
    p = Ptr{Cvoid}[0]
    dnnAllocateBuffer_F32(p, p_workspace)
    p_buf = p[1]

    dims = ntuple(i -> (size(x,i)+2padding[i]-ksize[i]) ÷ strides[i] + 1, N)
    y = Array{T}(dims..., size(x,3), size(x,4))
    resources = Ptr{Cvoid}[pointer(x),pointer(y),0,0,0,0,0,p_buf]
    dnnExecute(p_pool, resources)

    dnnPrimitiveAttributesDestroy(attr)
    dnnLayoutDelete_F32(layout)
    dnnDelete(p_pool)
    y
end
