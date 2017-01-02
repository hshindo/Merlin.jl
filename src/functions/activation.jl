export relu, clipped_relu, sigmoid
import Base.tanh

"""
    relu(x::Var)

Rectifier linear unit.
"""
function relu(x::Var)
    isvoid(x.data) && return Var(nothing, relu, (x,))
    y = relu(x.data)
    df(gy) = isvoid(x.grad) || ∇relu!(y, gy, x.data, x.grad)
    Var(y, df, (x,))
end

function relu{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    y
end

function ∇relu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(x[i]>T(0), gy[i], T(0))
    end
end

"""
    clipped_relu(x::Var)
"""
function clipped_relu(x::Var)
    isvoid(x.data) && return Var(nothing, clipped_relu, (x,))
    y = clipped_relu(x.data)
    df(gy) = isvoid(x.grad) || ∇clipped_relu!(y, gy, x.data, x.grad)
    Var(y, df, (x,))
end

function clipped_relu{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end
    y
end

function ∇clipped_relu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(T(0)<x[i]<T(20), gy[i], T(0))
    end
end

"""
    sigmoid(x::Var)
"""
function sigmoid(x::Var)
    isvoid(x.data) && return Var(nothing, sigmoid, (x,))
    y = sigmoid(x.data)
    df(gy) = isvoid(x.grad) || ∇sigmoid!(y, gy, x.data, x.grad)
    Var(y, df, (x,))
end

function sigmoid{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    y
end

function ∇sigmoid!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

"""
    tanh(x::Var)
"""
function tanh(x::Var)
    isvoid(x.data) && return Var(nothing, tanh, (x,))
    y = tanh(x.data)
    df(gy) = isvoid(x.grad) || ∇tanh!(y, gy, x.data, x.grad)
    Var(y, df, (x,))
end

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
