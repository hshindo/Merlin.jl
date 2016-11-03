export relu

"""
    relu(x::Var)
"""
function relu(vx::Var)
    y = relu(vx.data)
    Var(y, [vx], ∇relu!)
end

function relu{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = max(x[i], T(0))
    end
    y
end

function clipped_relu{T}(x::Array{T})
    y = similar(x)
    @inbounds @simd for i = 1:length(x)
        y[i] = min(max(x[i],T(0)), T(20))
    end
    y
end

relu(x::CuArray) = CUDNN.activation!(CUDNN_ACTIVATION_RELU, x)

function ∇relu!{T,N}(vy::Var{Array{T,N}})
    y = vy.data
    x, gx = vx.data, vx.grad
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
