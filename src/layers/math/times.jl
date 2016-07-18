import Base: .*, *

type ElemTimes <: Var
    data
    grad
    tails::Vector{Var}
end

type Times <: Var
    data
    grad
    tails::Vector{Var}
end

function .*(x1::Var, x2::Var)
    y = (hasdata(x1) && hasdata(x2)) ? x1.data .* x2.data : nothing
    ElemTimes(y, nothing, [x1,x2])
end
@compat (::ElemTimes)(x1::Var, x2::Var) = x1 .* x2

function *(x1::Var, x2::Var)
    y = (hasdata(x1) && hasdata(x2)) ? x1.data * x2.data : nothing
    Times(y, nothing, [x1,x2])
end
@compat (::Times)(x1::Var, x2::Var) = x1 * x2

function backward!(v::ElemTimes)
    hasgrad(v[1]) && ∇elemtimes!(v[2].data, v[1].grad, v.grad)
    hasgrad(v[2]) && ∇elemtimes!(v[1].data, v[2].grad, v.grad)
end

function ∇elemtimes!{T,N}(x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
    if length(gx1) < length(gy)
        @inbounds for k = 1:length(gx1):length(gy)
            @simd for i = 1:length(gx1)
                gx1[i] += gy[k+i-1] * x2[k+i-1]
            end
        end
    else
        broadcast!(.+, gx1, gx1, gy.*x2)
    end
end

backward!(v::Times) = ∇times!(v[1], v[2], v)

function ∇times!(x1::Var, x2::Var, y::Var)
    T = eltype(y.data)
    hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), y.grad, x2.data, T(1), x1.grad)
    hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.data, y.grad, T(1), x2.grad)
end
