import ..Merlin: relu, clipped_relu, sigmoid
import Base.tanh

function activation(x::Var, mode)
    desc = CUDNN.ActivationDesc(mode)
    y = CUDNN.activation(desc, x.data)
    df(gy) = CUDNN.∇activation!(desc, y, gy, x.data, x.grad, beta=1.0)
    Var(y, df, (x,))
end

relu{X<:CuArray}(x::Var{X}) = activation(x, CUDNN_ACTIVATION_RELU)

clipped_relu{X<:CuArray}(x::Var{X}) = activation(x, CUDNN_ACTIVATION_CLIPPED_RELU)

sigmoid{X<:CuArray}(x::Var{X}) = activation(x, CUDNN_ACTIVATION_SIGMOID)

tanh{X<:CuArray}(x::Var{X}) = activation(x, CUDNN_ACTIVATION_TANH)
