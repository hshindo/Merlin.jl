function activation{T}(x::CuArray{T}, mode)
    h = CUDNN.handle(x)
    desc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    cudnnActivationForward(h, desc, T[1], xdesc, x, T[0], xdesc, y)
    function backward!(v::Var)
        isvoid(v[1].grad) && return
        y, dy, x, dx = v.data, v.grad, v[1].data, v[1].grad
        cudnnActivationBackward(h, desc, T[1], xdesc, y, xdesc, dy, xdesc, x, T[1], xdesc, dx)
    end
    y, backward!
end

forward(::typeof(relu), x::CuArray) = activation(x, CUDNN_ACTIVATION_RELU)
forward(::typeof(clipped_relu), x::CuArray) = activation(x, CUDNN_ACTIVATION_CLIPPED_RELU)
forward(::typeof(sigmoid), x::CuArray) = activation(x, CUDNN_ACTIVATION_SIGMOID)
forward(::typeof(tanh), x::CuArray) = activation(x, CUDNN_ACTIVATION_TANH)
