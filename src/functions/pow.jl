import Base: .^

"""
    .^(x::Var, a::Number)
"""
function .^(x::Var, a::Number)
    y = x.data .^ a
    df(gy) = hasgrad(x) && ∇elemexp!(a, x.data, x.grad, y, gy)
    Var(y, [x], .^, df)
end

function ∇elempow!{T}(a, x::Array{T}, gx::Array{T}, y::Array{T}, gy::Array{T})
   @inbounds @simd for i = 1:length(gx)
       gx[i] += gy[i] * T(a) * y[i] / x[i]
   end
   gx
end

#=
"""
    \.\/(x1::Var, x2::Var)
"""
function ./(x1::Var, x2::Var)
    y = x1.data ./ x2.data
    function df{T}(gy::Array{T})
        hasgrad(x1) && ∇elemtimes!(T(1)./x2.data, x1.grad, gy)
        hasgrad(x2) && ∇elemexp!(-1, x2.data, x2.grad, y, gy)
    end
    Var(y, [x1,x2], ./, df)
end
=#
