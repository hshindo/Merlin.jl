export relu, clipped_relu, sigmoid
import Base.tanh

"""
    relu(x::Var)

Rectifier linear unit.
"""
function relu(x::Var)
    y, backward! = forward0(relu, x)
    Var(y, (relu,x), backward!)

    isvoid(x.data) && return Var(nothing, (relu,x))
    
end

function forward{T}(::typeof(relu), x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    backward!(gy, gx) = isvoid(gx) || ∇relu!(y, gy, x, gx)
    y, backward!
end

function ∇relu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end

"""
    clipped_relu(x::Var)
"""
clipped_relu(x::Var) = forward0(clipped_relu, x)

function forward{T}(::typeof(clipped_relu), x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end
    backward!(gy, gx) = isvoid(gx) || ∇clipped_relu!(y, gy, x, gx)
    y, backward!
end

function ∇clipped_relu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(T(0) < x[i] < T(20), gy[i], T(0))
    end
end

"""
    sigmoid(x::Var)
"""
sigmoid(x::Var) = forward0(sigmoid, x)

function forward{T}(::typeof(sigmoid), x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    backward!(gy, gx) = isvoid(gx) || ∇sigmoid!(y, gy, x, gx)
    y, backward!
end

function ∇sigmoid!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

"""
    tanh(x::Var)
"""
tanh(x::Var) = forward0(tanh, x)

function forward{T}(::typeof(tanh), x::Array{T})
    y = tanh(x)
    backward!(gy, gx) = isvoid(gx) || ∇tanh!(y, gy, x, gx)
    y, backward!
end

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
