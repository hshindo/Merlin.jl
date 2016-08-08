import Base: +, -, *

type Plus
  as::Vector
end

+(x1::Var, x2::Var) = plus([x1,x2], [1,1])
+(a::Number, x::Var) = Var(a) + x
+(x::Var, a::Number) = x + Var(a)

-(x1::Var, x2::Var) = plus([x1,x2], [1,-1])
-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)
-(x::Var) = plus([x], [-1])

*(a::Number, x::Var) = plus([x], [a])
*(x::Var, a::Number) = a * x

function +(a1, x1::Var, a2, x2::Var)
    
end

function plus(xs::Vector, as::Vector)
  (hasdata(xs[1]) && hasdata(xs[2])) || return Plus(nothing, nothing, xs, as)
  maxi, maxlen = 0, 0
  for i = 1:length(xs)
    n = length(xs[i].data)
    n <= maxlen && continue
    maxi = i
    maxlen = n
  end
  y = zeros(xs[maxi].data)
  T = eltype(y)
  for i = 1:length(xs)
    add!(as[i], xs[i].data, y)
  end
  Plus(y, nothing, xs, as)
end

function backward!(v::Plus)
  xs = v.tails
  for i = 1:length(xs)
    hasgrad(xs[i]) && ∇add!(v.as[i], xs[i].grad, v.grad)
  end
end

function add!{T}(a::Number, x::Array{T}, y::Array{T})
  n = length(x)
  for k = 1:n:length(y)
    BLAS.axpy!(n, T(a), pointer(x), 1, pointer(y,k), 1)
  end
end
add!{T}(a::Number, x::Number, y::Array{T}) = broadcast!(+, y, y, x)

function ∇add!{T}(a::Number, gx::Array{T}, gy::Array{T})
  n = length(gx)
  for k = 1:n:length(gy)
    BLAS.axpy!(n, T(a), pointer(gy,k), 1, pointer(gx), 1)
  end
end

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
