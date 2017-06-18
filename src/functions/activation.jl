export relu, clipped_relu, sigmoid
import Base.tanh

"""
    relu(x::Var)
"""
function relu(x::Var)
    y = Var(relu(x.data), x.batchdims, relu, (x,))
    y.df! = () -> begin
        isvoid(x.grad) || ∇relu!(y.grad, x.data, x.grad)
    end
    y
end

function relu(x::Array{T}) where T
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    y
end

function ∇relu!(gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end

"""
    clipped_relu(x::Var)
"""
function clipped_relu(x::Var)
    y = Var(clipped_relu(x.data), x.batchdims, clipped_relu, (x,))
    y.df! = () -> begin
        isvoid(x.grad) || ∇clipped_relu!(y.grad, x.data, x.grad)
    end
    y
end

function clipped_relu!(x::Array{T}) where T
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end
    y
end

function ∇clipped_relu!(gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(x)
        gx[i] += ifelse(T(0) < x[i] < T(20), gy[i], T(0))
    end
end

"""
    sigmoid(x::Var)
"""
function sigmoid(x::Var)
    y = Var(sigmoid(x.data), x.batchdims, sigmoid, (x,))
    y.df! = () -> begin
        isvoid(x.grad) || ∇sigmoid!(y.data, y.grad, x.data, x.grad)
    end
    y
end

function sigmoid(x::Array{T}) where T
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    y
end

function ∇sigmoid!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

"""
    tanh(x::Var)
"""
function tanh(x::Var)
    y = Var(tanh.(x.data), x.batchdims, tanh, (x,))
    y.df! = () -> begin
        isvoid(x.grad) || ∇tanh!(y.data, y.grad, x.data, x.grad)
    end
    y
end

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
