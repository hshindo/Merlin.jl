import Merlin: relu, ∇relu!

relu(x::CuArray) = CUDNN.activation(CUDNN_ACTIVATION_RELU, x)

function ∇relu!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, gy, x, gx, beta=1.0)
end
