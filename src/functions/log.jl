import Base.log

"""
    log
"""
function log(x::Var)
    y = log(x.data)
    df(gy) = hasgrad(x) && (x.grad = ∇log!(x.data, x.grad, gy))
    Var(y, [x], log, df)
end

function ∇log!{T}(x::Array{T}, gx::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] / x[i]
    end
    gx
end

∇log!{T<:Number}(x::T, gx::T, gy::T) = gx + gy / x
