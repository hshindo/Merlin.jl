export relu

"""
    relu(x::Var)
"""
@graph function relu(x::Var)
    y = relu(x.data)
    df(gy) = isconst(x) || ∇relu!(x.data, x.grad, y, gy)
    Var(y, [x], df)
end

function relu{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    y
end

relu(x::CuArray) = CUDNN.activation!(CUDNN_ACTIVATION_RELU, x, similar(x))

∇relu!(y::Var) = ∇relu!(y[1].data, y[1].grad, y.data, y.grad)

function ∇relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end

function ∇relu!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, gy, x, gx, beta=1.0)
end
