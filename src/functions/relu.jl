export relu

"""
    relu(x::Var)
"""
function relu(x::Var)
    y = Var(eltype(x), size(x), (x,))
    relu!(x.data, y.data)
    y.df = () -> ∇relu!(y.data, y.grad, x.data, x.grad)
    y
end

function relu!{T}(x::Array{T}, y::Array{T})
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    y
end

function clipped_relu!{T}(x::Array{T}, y::Array{T})
    @inbounds @simd for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end
    y
end

relu(x::CuArray) = CUDNN.activation!(CUDNN_ACTIVATION_RELU, x)

function ∇relu!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(x[i] > T(0), gy[i], T(0))
    end
end

function ∇clipped_relu!{T}(x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(x)
        gx[i] += ifelse(T(0)<x[i]<T(20), gy[i], T(0))
    end
end

function ∇relu!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_RELU, y, gy, x, gx, beta=1.0)
end
