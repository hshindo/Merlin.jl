import Base.exp

"""
    exp
"""
@graph function exp(vx::Var)
    y = exp(vx.value)
    df(y::Var) = ∇exp!(x, y)
    Var(y, df)
end
exp(vx::Var{Void}) = Var(nothing, [vx], exp, nothing)

function exp{A<:Array}(x::Var{A})
    y = exp(x.value)
    df(out::Var) = ∇exp!(out.value, x.value)
    Var(y, df)
end

function ∇exp!{A<:Array}(x::Var{A}, y::Var{A})
    @inbounds @simd for i = 1:length(x.value)
        x.grad[i] += y.grad[i] * y.value[i]
    end
end

function ∇exp!{T}(gx::Array{T}, y::Array{T}, gy::Array{T})
    @inbounds @simd for i = 1:length(gx)
        gx[i] += gy[i] * y[i]
    end
    gx
end

#=
@graph function exp(x::Var)
    y = exp(x.data)
    df(gy) = isconst(x) || (x.grad = ∇exp!(x.grad, y, gy))
    Var(y, [x], exp, df)
end



∇exp!{T<:Number}(gx::T, y::T, gy::T) = gx + gy * y
=#
