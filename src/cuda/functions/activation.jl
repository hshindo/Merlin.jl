function activation!(out::Var, x::CuArray{T}, mode) where T
    h = CUDNN.handle(x)
    actdesc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    CUDNN.cudnnActivationForward(h, actdesc, T[1], xdesc, x, T[0], xdesc, y)

    out.data = y
    out.df! = () -> begin
        isvoid(out[1].grad) && return
        gy, gx = out.grad, out[1].grad
        CUDNN.cudnnActivationBackward(h, actdesc, T[1], xdesc, y, xdesc, gy, xdesc, x, T[1], xdesc, gx)
    end
end
relu!(out::Var, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_RELU)
clipped_relu!(out::Var, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU)
sigmoid!(out::Var, x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_SIGMOID)
tanh!(out::Var, x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_TANH)
