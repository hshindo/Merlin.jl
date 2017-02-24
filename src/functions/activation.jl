export relu, clipped_relu, sigmoid
import Base.tanh

function activation{T}(x::CuArray{T}, mode)
    h = CUDNN.handle(x)
    desc = CUDNN.ActivationDesc(mode)
    y = similar(x)
    xdesc = CUDNN.TensorDesc(x)
    cudnnActivationForward(h, desc, T[1], xdesc, x, T[0], xdesc, y)
    function backward!(gy, gx)
        isvoid(gx) && return
        cudnnActivationBackward(h, desc, T[1], xdesc, y, xdesc, gy, xdesc, x, T[1], xdesc, gx)
    end
    y, backward!
end

"""
    relu(x::Var)

Rectifier linear unit.
"""
@forward relu(x::Var)

function forward{T}(::typeof(relu), x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    backward!(gy, gx) = isvoid(gx) || ∇relu!(y, gy, x, gx)
    y, backward!
end

forward(::typeof(relu), x::CuArray) = activation(x, CUDNN_ACTIVATION_RELU)

function ∇relu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end

"""
    clipped_relu(x::Var)
"""
@forward clipped_relu(x::Var)

function forward{T}(::typeof(clipped_relu), x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end
    backward!(gy, gx) = isvoid(gx) || ∇clipped_relu!(y, gy, x, gx)
    y, backward!
end

forward(::typeof(clipped_relu), x::CuArray) = activation(x, CUDNN_ACTIVATION_CLIPPED_RELU)

function ∇clipped_relu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(T(0) < x[i] < T(20), gy[i], T(0))
    end
end

"""
    sigmoid(x::Var)
"""
@forward sigmoid(x::Var)

function forward{T}(::typeof(sigmoid), x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    backward!(gy, gx) = isvoid(gx) || ∇sigmoid!(y, gy, x, gx)
    y, backward!
end

forward(::typeof(sigmoid), x::CuArray) = activation(x, CUDNN_ACTIVATION_SIGMOID)

function ∇sigmoid!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

"""
    tanh(x::Var)
"""
@forward tanh(x::Var)

function forward{T}(::typeof(tanh), x::Array{T})
    y = tanh(x)
    backward!(gy, gx) = isvoid(gx) || ∇tanh!(y, gy, x, gx)
    y, backward!
end

forward(::typeof(tanh), x::CuArray) = activation(x, CUDNN_ACTIVATION_TANH)

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
