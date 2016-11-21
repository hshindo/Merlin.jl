export sigmoid

"""
    sigmoid(x::Var)
"""
function sigmoid(x::Var)
    x.data == nothing && return Var(nothing, (sigmoid,x))
    y = Var(eltype(x), size(x), (x,))
    sigmoid!(x.data, y.data)
    y.df = () -> isconst(x) || ∇sigmoid!(y.data, y.grad, x.data, x.grad)
    y
end

function sigmoid!{T}(x::Array{T}, y::Array{T})
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
end

sigmoid!(x::CuArray, y::CuArray) = CUDNN.activation(CUDNN_ACTIVATION_SIGMOID, x)

function ∇sigmoid!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

function ∇sigmoid!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_SIGMOID, y, gy, x, gx, beta=1.0)
end
