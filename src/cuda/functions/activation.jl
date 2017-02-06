import ..Merlin: relu, clipped_relu, sigmoid
import Base.tanh

function activation{T}(x::CuArray{T}, mode)
    h = CUDNN.handle(x)
    desc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    cudnnActivationForward(h, desc, T[1], xdesc, x, T[0], xdesc, y)
    function backward!(dy, dx)
        cudnnActivationBackward(h, desc, T[1], xdesc, y, xdesc, dy, xdesc, x, T[1], xdesc, dx)
    end
    y, backward!
end

forward(::typeof(relu), x::CuArray) = activation(x, CUDNN_ACTIVATION_RELU)

forward(::typeof(clipped_relu), x::CuArray) = activation(x, CUDNN_ACTIVATION_CLIPPED_RELU)

forward(::typeof(sigmoid), x::CuArray) = activation(x, CUDNN_ACTIVATION_SIGMOID)

forward(::typeof(tanh), x::CuArray) = activation(x, CUDNN_ACTIVATION_TANH)
