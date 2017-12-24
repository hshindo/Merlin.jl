function softmax!(out, x::CuArray{T,N}, algo=CUDNN.CUDNN_SOFTMAX_ACCURATE) where {T,N}
    @assert 1 <= N <= 4
    h = CUDNN.HANDLE
    xdesc = CUDNN.TensorDesc(x, pad=4-N)
    # dims1 = ntuple(_ -> 1, 4-ndims(x))
    # x4d = reshape(x, dims1..., size(x)...)
    y = similar(x)
    mode = CUDNN.CUDNN_SOFTMAX_MODE_CHANNEL
    CUDNN.cudnnSoftmaxForward(h, algo, mode, T[1], xdesc, x, T[0], xdesc, y)

    out.data = y
    out.∇! = () -> begin
        isvoid(out[1].grad) || CUDNN.cudnnSoftmaxBackward(h, algo, mode, T[1], xdesc, out.data, xdesc, out.grad, T[1], xdesc, out[1].grad)
    end
    out
end

softmax(x::CuArray) = softmax!(Var(),x,CUDNN.CUDNN_SOFTMAX_ACCURATE).data
logsoftmax(x::CuArray) = softmax!(Var(),x,CUDNN_SOFTMAX_LOG).data
