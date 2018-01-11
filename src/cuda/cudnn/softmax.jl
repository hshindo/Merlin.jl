# cudnnSoftmaxAlgorithm_t
const CUDNN_SOFTMAX_FAST = Cint(0)
const CUDNN_SOFTMAX_ACCURATE = Cint(1)
const CUDNN_SOFTMAX_LOG = Cint(2)

# cudnnSoftmaxMode_t
const CUDNN_SOFTMAX_MODE_INSTANCE = Cint(0)
const CUDNN_SOFTMAX_MODE_CHANNEL = Cint(1)

function softmax(x::CuArray{T,N}, algo::Cint) where {T,N}
    @assert N <= 2
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    dims = [1, 1, size(x,1), size(x,2)]
    xdesc = TensorDesc(T, dims)
    y = similar(x)
    @cudnn(:cudnnSoftmaxForward,
        (Cptr,Cint,Cint,
        Cptr,Cptr,Cptr,
        Cptr,Cptr,Cptr),
        gethandle(), algo, mode, T[1], xdesc, x, T[0], xdesc, y)
    y
end

function ∇softmax!(y::CuArray{T}, dy, dx, algo) where T
    dims = [1, 1, size(y,1), size(y,2)]
    ydesc = TensorDesc(T, dims)
    mode = CUDNN_SOFTMAX_MODE_CHANNEL
    @cudnn(:cudnnSoftmaxBackward,
        (Cptr,Cint,Cint,
        Cptr,Cptr,Cptr,Cptr,Cptr,
        Cptr,Cptr,Cptr),
        gethandle(), algo, mode, T[1], ydesc, y, ydesc, dy, T[1], ydesc, dx)
end
