export BatchNorm

mutable struct BatchNorm <: Functor
    scale::Var
    bias::Var
    running_mean::Var
    running_var::Var
end

function BatchNorm(::Type{T}, inchannel::Int) where T
    scale = ones(T, 1, 1, inchannel, 1)
    bias = zero(scale)
    running_mean = zero(scale)
    running_var = zero(scale)
    BatchNorm(parameter(scale), parameter(bias), parameter(running_mean), parameter(running_var))
end

function (f::BatchNorm)(x::Var)
    ydata, work = f(x.data)
    Var(ydata, ∇batchnorm!, (f,x,work))
end

function (f::BatchNorm)(x::CuMatrix)
    mode = CUDNN.CUDNN_BATCHNORM_PER_ACTIVATION
    x = reshape(x, 1, 1, size(x,1), size(x,2))
    y, work = CUDNN.batchnorm!(mode, x, f.scale.data, f.bias.data, f.running_mean.data, f.running_var.data, training=istraining())
    y = reshape(y, size(y,3), size(y,4))
    y, work
end

function ∇batchnorm!(y::Var, f::BatchNorm, x::Var, work)
    ∇batchnorm!(y.grad, f, x.data, x.grad, work)
end

function ∇batchnorm!(gy::CuArray, f::BatchNorm, x::CuArray, gx::CuArray, work)
    CUDNN.∇batchnorm!(work, gy, gx, f.scale.grad, f.bias.grad)
end
