export sigmoid

"""
    sigmoid(x::Var)
"""
@graph function sigmoid(x::Var)
    y = sigmoid(x.data)
    df(gy) = isconst(x) || ∇sigmoid!(x.data, x.grad, y, gy)
    Var(y, [x], df)
end

function sigmoid{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = 1 / (1 + exp(-x[i]))
    end
    y
end

sigmoid(x::CuArray) = CUDNN.activation!(CUDNN_ACTIVATION_SIGMOID, x, similar(x))

function ∇sigmoid!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i] * (T(1) - y[i])
    end
end

function ∇sigmoid!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_SIGMOID, y, gy, x, gx, beta=1.0)
end
