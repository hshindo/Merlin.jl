import Base.tanh
import Merlin: ∇tanh!

tanh(x::CuArray) = CUDNN.activation(CUDNN_ACTIVATION_TANH, x)

function ∇tanh!(y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, gy, x, gx, beta=1.0)
end
