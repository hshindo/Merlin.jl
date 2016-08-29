import Base: *, .*

"""
    \*(x1::Var, x2::Var)
    \*(a::Number, x::Var)
    \*(x::Var, a::Number)
"""
function *(x1::Var, x2::Var)
    y = x1.data * x2.data
    function df{T}(gy::UniArray{T})
        hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), gy, x2.data, T(1), x1.grad)
        hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.data, gy, T(1), x2.grad)
    end
    Var(y, [x1,x2], *, df)
end
*(a::Number, x::Var) = axsum([a], [x])
*(x::Var, a::Number) = a * x

"""
    \.\*(x1::Var, x2::Var)
"""
function .*(x1::Var, x2::Var)
    y = x1.data .* x2.data
    function df(gy)
        hasgrad(x1) && ∇elemtimes!(x2.data, x1.grad, gy)
        hasgrad(x2) && ∇elemtimes!(x1.data, x2.grad, gy)
    end
    Var(y, [x1,x2], .*, df)
end

function ∇elemtimes!{T}(x2::UniArray{T}, gx1::UniArray{T}, gy::UniArray{T})
    if length(gx1) < length(gy)
        @inbounds for k = 0:length(gx1):length(gy)-1
            @simd for i = 1:length(gx1)
                gx1[i] += gy[i+k] * x2[i+k]
            end
        end
    else
        @inbounds for k = 0:length(x2):length(gy)-1
            @simd for i = 1:length(x2)
                gx1[i+k] += gy[i+k] * x2[i]
            end
        end
    end
end
