import Merlin: clipped_relu, relu, sigmoid
import Base.tanh

function activation{T<:CuArray}(x::Var{T}, mode)
    desc = CUDNN.ActivationDesc(mode)
    y = CUDNN.activation(desc, x)
    df(gy::CuArray) = CUDNN.∇activation!(desc, y, gy, x, gx, beta=1.0)
    Var(y, df, (x,))
end

clipped_relu{T<:CuArray}(x::Var{T}) = activation(x, CUDNN_ACTIVATION_CLIPPED_RELU)

relu{T<:CuArray}(x::Var{T}) = activation(x, CUDNN_ACTIVATION_RELU)

sigmoid{T<:CuArray}(x::Var{T}) = CUDNN.activation(x, CUDNN_ACTIVATION_SIGMOID)

tanh{T<:CuArray}(x::Var{T}) = CUDNN.activation(x, CUDNN_ACTIVATION_TANH)
