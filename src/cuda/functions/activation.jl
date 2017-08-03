function activation!{T}(out::Var, x::CuArray{T}, mode)
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

function activation!(out::Var, x::CuArray, mode)
    h = CUDNN.handle(x)
    actdesc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    CUDNN.cudnnActivationForward(h, actdesc, T[1], xdesc, x, T[0], xdesc, y)
    Var(y, activation, (x,mode), work=(h,actdesc,xdesc))
end

function _activation(x::CuArray, mode)
    h = CUDNN.handle(x)
    actdesc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    CUDNN.cudnnActivationForward(h, actdesc, T[1], xdesc, x, T[0], xdesc, y)
    y, h, actdesc, xdesc
end
activation(x::CuArray, mode) = _activation(x,mode)[1]

function gradient!(::typeof(activation), y::Var)
    h, actdesc, xdesc = y.work
    x, mode = y.args
    isvoid(x.grad) && return
    CUDNN.cudnnActivationBackward(h, actdesc, T[1], xdesc, y.data, xdesc, y.grad, xdesc, x.data, T[1], xdesc, x.grad)
end
