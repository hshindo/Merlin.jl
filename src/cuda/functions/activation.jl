function activation!(out, x::CuArray{T}, mode) where T
    h = CUDNN.HANDLE
    actdesc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    CUDNN.cudnnActivationForward(h, actdesc, T[1], xdesc, x, T[0], xdesc, y)

    out.data = y
    out.∇! = () -> begin
        if !isvoid(out[1].grad)
            CUDNN.cudnnActivationBackward(h, actdesc, T[1], xdesc, out.data, xdesc, out.grad, xdesc, out[1].data, T[1], xdesc, out[1].grad)
        end
    end
    out
end
clipped_relu!(out, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU)
elu!(out, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_ELU)
relu!(out, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_RELU)
sigmoid!(out, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_SIGMOID)
tanh!(out, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_TANH)

activation(x::CuArray, mode) = activation!(Var(),x,mode).data
clipped_relu(x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU)
elu(x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_ELU)
relu(x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_RELU)
sigmoid(x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_SIGMOID)
Base.tanh(x::CuArray) = activation(x, CUDNN.CUDNN_ACTIVATION_TANH)
