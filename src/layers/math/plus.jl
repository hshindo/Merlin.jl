import Base: +, -, *

type Plus <: Var
  data
  grad
  tails::Vector
  as::Vector
end

@compat (::Plus)(x1::Var, x2::Var) = x1 + x2

+(x1::Var, x2::Var) = plus([x1,x2], [1,1])
+(a::Number, x::Var) = Data(a) + x
+(x::Var, a::Number) = x + Data(a)

-(x1::Var, x2::Var) = plus([x1,x2], [1,-1])
-(a::Number, x::Var) = Data(a) - x
-(x::Var, a::Number) = x - Data(a)
-(x::Var) = plus([x], [-1])

*(a::Number, x::Var) = plus([x], [a])
*(x::Var, a::Number) = a * x

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
