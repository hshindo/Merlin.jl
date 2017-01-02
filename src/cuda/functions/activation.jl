import ..Merlin: relu, clipped_relu, sigmoid
import Base.tanh

function activation(x::Var, mode)
    desc = CUDNN.ActivationDesc(mode)
    y = CUDNN.activation(desc, x)
    df(gy::CuArray) = CUDNN.∇activation!(desc, y, gy, x, gx, beta=1.0)
    Var(y, df, (x,))
end

clipped_relu(x::Var) = activation(x, CUDNN_ACTIVATION_CLIPPED_RELU)

relu(x::Var) = activation(x, CUDNN_ACTIVATION_RELU)

sigmoid(x::Var) = CUDNN.activation(x, CUDNN_ACTIVATION_SIGMOID)

tanh(x::Var) = CUDNN.activation(x, CUDNN_ACTIVATION_TANH)
