export relu, clipped_relu, sigmoid
import Base.tanh

"""
    relu(x::Var)
"""
function relu(x::Var)
    y = Var(nothing, relu, (x,))
    isvoid(x.data) || relu!(y, x.data)
    y
end

function relu!(out::Var, x::Array{T}) where T
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end

    out.data = y
    out.df! = () -> begin
        isvoid(out[1].grad) && return
        ∇relu!(out.grad, out[1].data, out[1].grad)
    end
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
    y = Var(nothing, clipped_relu, (x,))
    isvoid(x.data) || clipped_relu!(y, x.data)
    y
end

function clipped_relu!(out::Var, x::Array{T}) where T
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end

    out.data = y
    out.df! = () -> begin
        isvoid(out[1].grad) && return
        ∇clipped_relu!(out.grad, out[1].data, out[1].grad)
    end
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
    y = Var(nothing, sigmoid, (x,))
    isvoid(x.data) || sigmoid!(y, x.data)
    y
end

function sigmoid!(out::Var, x::Array{T}) where T
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end

    out.data = y
    out.df! = () -> begin
        isvoid(out[1].grad) && return
        ∇sigmoid!(out.data, out.grad, out[1].data, out[1].grad)
    end
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
    y = Var(nothing, tanh, (x,))
    isvoid(x.data) || tanh!(y, x.data)
    y
end

function tanh!(out::Var, x::Array{T}) where T
    out.data = tanh.(x)
    out.df! = () -> begin
        isvoid(out[1].grad) && return
        ∇tanh!(out.data, out.grad, out[1].data, out[1].grad)
    end
end

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
