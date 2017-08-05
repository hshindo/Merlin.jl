export relu, clipped_relu, sigmoid
import Base.tanh

"""
    relu(x::Var)
"""
function relu(x::Var)
    Var(relu(x.data), x.batchdims, relu, (x,))
end
relu(x::Node) = Node(relu, x)

function relu(x::Array{T}) where {T}
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    y
end

function addgrad!(y::Var, ::typeof(relu), x::Var)
    isvoid(x.grad) && return
    ∇relu!(y.grad, x.data, x.grad)
end

function ∇relu!(gy::Array{T}, x::Array{T}, gx::Array{T}) where {T}
    @inbounds for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end

"""
    clipped_relu(x::Var)
"""
function clipped_relu(x::Var)
    Var(clipped_relu(x.data), x.batchdims, clipped_relu, (x,))
end
clipped_relu(x::Node) = Node(clipped_relu, x)

function clipped_relu(x::Array{T}) where {T}
    y = similar(x)
    @inbounds for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end
    y
end

function addgrad!(y::Var, ::typeof(clipped_relu), x::Var)
    isvoid(x.grad) && return
    ∇clipped_relu!(y.grad, x.data, x.grad)
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
    Var(sigmoid(x.data), x.batchdims, sigmoid, (x,))
end
sigmoid(x::Node) = Node(sigmoid, x)

function sigmoid(x::Array{T}) where {T}
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    y
end

function addgrad!(y::Var, ::typeof(sigmoid), x::Var)
    isvoid(x.grad) && return
    ∇sigmoid!(y.data, y.grad, x.data, x.grad)
end

function ∇sigmoid!(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T}) where {T}
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

"""
    tanh(x::Var)
"""
function tanh(x::Var)
    Var(tanh.(x.data), x.batchdims, tanh, (x,))
end
tanh(x::Node) = Node(tanh, x)

function addgrad!(y::Var, ::typeof(tanh), x::Var)
    isvoid(x.grad) && return
    ∇tanh!(y.data, y.grad, x.data, x.grad)
end

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
