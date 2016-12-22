export sigmoid

"""
    sigmoid(x::Var)
"""
sigmoid(x::Var{Void}) = Var(Void(), sigmoid, (x,))

function sigmoid(x::Var)
    y = sigmoid(x.data)
    df(gy) = isvoid(x.grad) || ∇sigmoid!(y, gy, x.data, x.grad)
    Var(y, df, (x,))
end

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
