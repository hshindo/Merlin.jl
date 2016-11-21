import Base.tanh

"""
    tanh(x::Var)
"""
function tanh(x::Var)
    x.data == nothing && return Var(nothing, (tanh,x))
    y = Var(eltype(x), size(x), (x,))
    tanh!(x.data, y.data)
    y.df = () -> isconst(x) || ∇tanh!(y.data, y.grad, x.data, x.grad)
    y
end

function tanh!{T}(x::Array{T}, y::Array{T})
    @inbounds @simd for i = 1:length(x)
        y[i] = tanh(x[i])
    end
end

tanh(x::CuArray) = CUDNN.activation(CUDNN_ACTIVATION_TANH, x)

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end

function ∇tanh!(x::CuArray, gx::CuArray, y::CuArray, gy::CuArray)
    CUDNN.∇activation!(CUDNN_ACTIVATION_TANH, y, gy, x, gx, beta=1.0)
end
