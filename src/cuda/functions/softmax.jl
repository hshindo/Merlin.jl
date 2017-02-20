import Merlin: softmax, logsoftmax

function forward{T,N}(::typeof(softmax), x::CuArray{T,N}; algo=CUDNN_SOFTMAX_ACCURATE)
    @assert 1 < N <= 4
    h = CUDNN.handle(x)
    pad = 4 - N
    xdesc = CUDNN.TensorDesc(x, pad=pad)
    y = similar(x)
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    cudnnSoftmaxForward(h, algo, mode, T[1], xdesc, x, T[0], xdesc, y)
    function backward!(dy, dx)
        cudnnSoftmaxBackward(h, algo, mode, T[1], xdesc, y, xdesc, dy, T[1], xdesc, dx)
    end
    y, backward!
end

forward(::typeof(logsoftmax), x::CuArray) = forward(softmax, x, algo=CUDNN_SOFTMAX_LOG)
