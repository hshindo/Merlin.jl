export relu, clipped_relu, sigmoid
import Base.tanh

"""
    relu(x::Var)
"""
function relu(x::Var)
    y = Var(relu(x.data), relu, (x,))
    y.df! = () -> begin
        isconst(x) && return
        ∇relu!(y.data, y.grad, x.data, x.grad)
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
relu(x::BatchedArray) = BatchedArray(relu(x.data), x.dims)

function ∇relu!(y, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end
∇relu!(y::BatchedArray, gy, x, gx) = ∇relu!(y.data, gy.data, x.data, gx.data)

"""
    clipped_relu(x::Var)
"""
function clipped_relu(x::Var)
    y = Var(clipped_relu(x.data), clipped_relu, (x,))
    y.df! = () -> begin
        isconst(x) && return
        ∇clipped_relu!(y.data, y.grad, x.data, x.grad)
    end
    y
end

function clipped_relu!{T}(out::Var, x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end
    y
end

function ∇clipped_relu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(T(0) < x[i] < T(20), gy[i], T(0))
    end
end

"""
    sigmoid(x::Var)
"""
function sigmoid(x::Var)
    y = Var(sigmoid(x.data), sigmoid, (x,))
    y.df! = () -> begin
        isconst(x) && return
        ∇sigmoid!(y.data, y.grad, x.data, x.grad)
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
sigmoid(x::BatchedArray) = BatchedArray(sigmoid(x.data), x.dims)

function ∇sigmoid!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where T
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end
∇sigmoid!(y::BatchedArray, gy, x, gx) = ∇sigmoid!(y.data, gy.data, x.data, gx.data)

"""
    tanh(x::Var)
"""
function tanh(x::Var)
    y = Var(tanh(x.data), tanh, (x,))
    y.df! = () -> begin
        isconst(x) && return
        ∇tanh!(y.data, y.grad, x.data, x.grad)
    end
    y
end

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
