export relu, clipped_relu

"""
    relu(x::Var)

Rectifier linear unit.
"""
function relu(x::Var)
    y = relu(x.data)
    df(gy) = isvoid(x.grad) || ∇relu!(y, gy, x.data, x.grad)
    Var(y, df, (x,))
end
relu(x::Var{Void}) = Var(Void(), relu, (x,))

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
function clipped_relu(x::Var)
    y = clipped_relu(x.data)
    df(gy) = isvoid(x.grad) || ∇clipped_relu!(y, gy, x.data, x.grad)
    Var(y, df, (x,))
end
clipped_relu(x::Var{Void}) = Var(Void(), clipped_relu, (x,))

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
