import Merlin: relu, clipped_relu, sigmoid
import Base: tanh
export activation

function activation(v::CuVar, mode)
    x = v.data
    h = CUDNN.handle(x)
    actdesc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    CUDNN.cudnnActivationForward(h, actdesc, T[1], xdesc, x, T[0], xdesc, y)
    CuVar(y, v.batchdims, activation, (v,mode), work=(h,actdesc,xdesc))
end

function activation(v::Var{<:CuArray{T}}, mode) where {T}
    x = v.data
    h = CUDNN.handle(x)
    actdesc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    CUDNN.cudnnActivationForward(h, actdesc, T[1], xdesc, x, T[0], xdesc, y)
    Var(y, v.batchdims, activation, (v,mode), work=(h,actdesc,xdesc))
end

relu(x::Var{<:CuArray}) = activation(x, CUDNN.CUDNN_ACTIVATION_RELU)
clipped_relu(x::Var{<:CuArray}) = activation!(x, CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU)
sigmoid(x::Var{<:CuArray}) = activation(x, CUDNN.CUDNN_ACTIVATION_SIGMOID)
tanh(x::Var{<:CuArray}) = activation(x, CUDNN.CUDNN_ACTIVATION_TANH)

function addgrad!(y::Var{<:CuArray{T}}, ::typeof(activation), x::Var, mode) where {T}
    isvoid(x.grad) && return
    h, actdesc, xdesc = y.work
    CUDNN.cudnnActivationBackward(h, actdesc, T[1], xdesc, y.data, xdesc, y.grad, xdesc, x.data, T[1], xdesc, x.grad)
end

function addgrad!(y::CuVar, ::typeof(activation), x::CuVar, mode)
    isvoid(x.grad) && return
    h, actdesc, xdesc = y.work
    CUDNN.cudnnActivationBackward(h, actdesc, T[1], xdesc, y.data, xdesc, y.grad, xdesc, x.data, T[1], xdesc, x.grad)
end
