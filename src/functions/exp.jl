import Base.exp

"""
    exp
"""
function exp(x::Var)
    x.data == nothing && return Var(nothing, exp, (x,))
    y = exp(x.data)
    df(gy) = ∇exp!(y, gy, x.grad)
    Var(y, df, (x,))
end

function ∇exp!{T}(y::Array{T}, gy::Array{T}, gx::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i]
    end
    gx
end
