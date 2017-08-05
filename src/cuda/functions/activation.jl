function activation!(out::Var, x::CuArray, mode)
    h = CUDNN.handle(x)
    actdesc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    CUDNN.cudnnActivationForward(h, actdesc, T[1], xdesc, x, T[0], xdesc, y)

    out.data = y
    out.work = h, actdesc, xdesc
end

function activation(var::Var{<:CuArray{T}}, mode) where {T}
    x = var.data
    h = CUDNN.handle(x)
    actdesc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    CUDNN.cudnnActivationForward(h, actdesc, T[1], xdesc, x, T[0], xdesc, y)
    Var(y, var.batchdims, relu, (var,), work=(h,actdesc,xdesc))
end

relu(var::Var{<:CuArray}) = activation(var, CUDNN.CUDNN_ACTIVATION_RELU)

function gradient!(y::Var{<:CuArray{T}}, ::typeof(relu)) where {T}
    x = y[1]
    h, actdesc, xdesc = y.work
    CUDNN.cudnnActivationBackward(h, actdesc, T[1], xdesc, y.data, xdesc, y.grad, xdesc, x.data, T[1], xdesc, x.grad)
end



relu!(out::Var, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_RELU)
clipped_relu!(out::Var, x::CuArray) = activation!(out, x, CUDNN.CUDNN_ACTIVATION_CLIPPED_RELU)
sigmoid!(out::Var, x::CuArray) = activation(out, x, CUDNN.CUDNN_ACTIVATION_SIGMOID)
tanh!(out::Var, x::CuArray) = activation(out, x, CUDNN.CUDNN_ACTIVATION_TANH)

function ∇activation!(out::Var, y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    h, actdesc, xdesc = out.work
    CUDNN.cudnnActivationBackward(h, actdesc, T[1], xdesc, y, xdesc, gy, xdesc, x, T[1], xdesc, gx)
end

function ∇relu!(out::Var, y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    ∇activation!(out, y, gy, x, gx)
end

function ∇clipped_relu!(out::Var, y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    ∇activation!(out, y, gy, x, gx)
end

function ∇sigmoid!(out::Var, y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    ∇activation!(out, y, gy, x, gx)
end

function ∇tanh!(out::Var, y::CuArray, gy::CuArray, x::CuArray, gx::CuArray)
    ∇activation!(out, y, gy, x, gx)
end
