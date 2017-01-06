export relu, clipped_relu, sigmoid
import Base.tanh

function activation{X<:Array}(x::Var{X}, f, ∇)
    y = f(x.data)
    df(gy) = isa(x.grad, Void) || ∇(y, gy, x.data, x.grad)
    Var(y, df, (x,))
end

"""
    relu(x::Var)

Rectifier linear unit.
"""
relu(x::Var) = activation(x, relu, ∇relu!)
relu(x::Var{Void}) = Var(nothing, relu, (x,))

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
clipped_relu(x::Var) = activation(x, clipped_relu, ∇clipped_relu!)
clipped_relu(x::Var{Void}) = Var(nothing, clipped_relu, (x,))

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
sigmoid(x::Var) = activation(x, sigmoid, ∇sigmoid!)
sigmoid(x::Var{Void}) = Var(nothing, sigmoid, (x,))

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
tanh(x::Var) = activation(x, tanh, ∇tanh!)
tanh(x::Var{Void}) = Var(nothing, tanh, (x,))

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
