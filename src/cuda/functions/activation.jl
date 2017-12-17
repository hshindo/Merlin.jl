function activation!(out::Var, x::CuArray, mode)
    h = CUDNN.handle(x)
    actdesc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    CUDNN.cudnnActivationForward(h, actdesc, T[1], xdesc, x, T[0], xdesc, y)

    out.data = y
    out.args = activation!, out.args[2], h, actdesc, xdesc
    out
end
relu!(out::Var, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_RELU)
clipped_relu!(out::Var, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU)
sigmoid!(out::Var, x::CuArray) = activation(out, x, CUDNN.CUDNN_ACTIVATION_SIGMOID)
tanh!(out::Var, x::CuArray) = activation(out, x, CUDNN.CUDNN_ACTIVATION_TANH)

activation(x::CuArray, mode) = activation!(Var(), x, mode).data
relu(x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_RELU)
clipped_relu(x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU)
sigmoid(x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_SIGMOID)
tanh(x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_TANH)

function addgrad!(y::Var, ::typeof(activation), x::Var, mode, h, actdesc, xdesc)
    T = eltype(y)
    isvoid(x.grad) || CUDNN.cudnnActivationBackward(h, actdesc, T[1], xdesc, y.data, xdesc, y.grad, xdesc, x.data, T[1], xdesc, x.grad)
end
