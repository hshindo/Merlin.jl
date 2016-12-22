import Base.tanh

"""
    tanh(x::Var)
"""
tanh(x::Var{Void}) = Var(Void(), tanh, (x,))

function tanh(x::Var)
    y = tanh(x.data)
    df(gy) = isvoid(x.grad) || ∇tanh!(y, gy, x.data, x.grad)
    Var(y, df, (x,))
end

function ∇tanh!{T}(y::Array{T}, gy::Array{T}, x::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * (T(1) - y[i] * y[i])
    end
end
