# cudnnSoftmaxAlgorithm_t
const CUDNN_SOFTMAX_FAST = Cint(0)
const CUDNN_SOFTMAX_ACCURATE = Cint(1)
const CUDNN_SOFTMAX_LOG = Cint(2)

# cudnnSoftmaxMode_t
const CUDNN_SOFTMAX_MODE_INSTANCE = Cint(0)
const CUDNN_SOFTMAX_MODE_CHANNEL = Cint(1)

function softmax(x::CuArray{T}; algo=CUDNN_SOFTMAX_ACCURATE, mode=CUDNN_SOFTMAX_MODE_CHANNEL) where T
    h = gethandle()
    xdesc = TensorDesc(x, 4)
    y = similar(x)
    @cudnn(:cudnnSoftmaxForward,
        (Cptr,Cint,Cint,
        Cptr,Cptr,Cptr,
        Cptr,Cptr,Cptr),
        h, algo, mode, T[1], xdesc, x, T[0], xdesc, y)
    y
end

function ∇softmax!(y::CuArray{T}, dy, dx, algo=CUDNN_SOFTMAX_ACCURATE) where T
    h = gethandle()
    ydesc = TensorDesc(y, 4)
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    @cudnn(:cudnnSoftmaxBackward,
        (Cptr,Cint,Cint,
        Cptr,Cptr,Cptr,Cptr,Cptr,
        Cptr,Cptr,Cptr),
        h, algo, mode, T[1], ydesc, y, ydesc, dy, T[1], ydesc, dx)
end
