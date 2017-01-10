import Merlin: softmax, logsoftmax, ∇softmax!, ∇logsoftmax!

softmax(x::CuArray) = CUDNN.softmax(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, x)

function ∇softmax!(y::CuArray, gy::CuArray, gx::CuArray)
    CUDNN.∇softmax!(CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, y, gy, gx; beta=1.0)
end

logsoftmax(x::CuArray) = CUDNN.softmax(CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, x)

function ∇logsoftmax!(y::CuArray, gy::CuArray, gx::CuArray)
    CUDNN.∇logsoftmax!(CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, y, gy, gx; beta=1.0)
end
