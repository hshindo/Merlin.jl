export relu

"""
    relu(x::Var)
"""
function relu(x::Var)
    y = relu(x.data)
    df(gy) = isconstant(x) || ∇relu!(x.data, x.grad, y, gy)
    Var(y, [x], relu, df)
end
relu(x::GraphNode) = GraphNode(relu, x)

function relu{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    y
end

relu(x::CuArray) = CUDNN.activation!(CUDNN_ACTIVATION_RELU, x, similar(x))

function ∇relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end

function ∇relu!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, gy, x, gx, beta=1.0)
end
