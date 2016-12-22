import Merlin: sigmoid, ∇sigmoid!

sigmoid(x::CuArray) = CUDNN.activation(CUDNN_ACTIVATION_SIGMOID, x)

function ∇sigmoid!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_SIGMOID, y, gy, x, gx, beta=1.0)
end
