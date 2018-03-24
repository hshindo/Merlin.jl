function (conv::Conv)(x::CuArray)
    CUDNN.convolution(conv.w.data, x, conv.pads, conv.strides, conv.dilations)
end

function ∇conv!(gy::CuArray, conv::Conv, x::CuArray, gx)
    w, gw = conv.w.data, conv.w.grad
    CUDNN.∇convolution!(gy, w, gw, x, gx, conv.pads, conv.strides, conv.dilations)
end

function (cuda::CUDABackend)(conv::Conv)
    Conv(cuda(conv.w), cuda(conv.b), conv.pads, conv.strides, conv.dilations)
end
