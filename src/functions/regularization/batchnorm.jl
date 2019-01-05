export BatchNorm

mutable struct BatchNorm <: Functor
    scale
    bias
    running_mean
    running_var
end

function BatchNorm()
    BatchNorm(nothing, nothing, nothing, nothing)
end

function (f::BatchNorm)(x::Var)
    if isnothing(f.scale)
        f.scale = parameter(zeros(eltype(x),1,1,size(x,1)))
        f.bias = parameter(zero(f.scale.data))
        f.running_mean = parameter(zero(f.scale.data))
        f.running_var = parameter(zero(f.scale.data))
    end
    ydata, work = f(x.data)
    Var(ydata, ∇batchnorm!, (f,x,work))
end

function (f::BatchNorm)(x::CuMatrix)
    mode = CUDNN.CUDNN_BATCHNORM_PER_ACTIVATION
    x = reshape(x, 1, 1, size(x,1), size(x,2))
    CUDNN.batchnorm(mode, x, f.scale, f.bias, f.running_mean, f.running_var, training=istraining())
end
