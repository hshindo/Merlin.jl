import Base: +, -, *

type Plus <: Layer
  as::Vector
  xs::Vector
  y
  gy
end

function Plus{T<:Layer}(as::Vector, xs::Vector{T})
  maxi, maxlen = 1, length(xs[1].y)
  for i = 2:length(xs)
    n = length(xs[i].y)
    n <= maxlen && continue
    maxi = i
    maxlen = n
  end
  y = zeros(xs[maxi].y)
  for i = 1:length(xs)
    add!(as[i], xs[i].y, y)
  end
  Plus(as, xs, y, nothing)
end

tails(l::Plus) = l.xs

+(x1::Layer, x2::Layer) = Plus([1,1], [x1,x2])
+(a::Number, x::Layer) = Data(a) + x
+(x::Layer, a::Number) = x + Data(a)

-(x1::Layer, x2::Layer) = Plus([1,-1], [x1,x2])
-(a::Number, x::Layer) = Data(a) - x
-(x::Layer, a::Number) = x - Data(a)
-(x::Layer) = Plus([-1], [x])

*(a::Number, x::Layer) = Plus([a], [x])
*(x::Layer, a::Number) = a * x

function backward!(l::Plus)
  xs = l.xs
  for i = 1:length(xs)
    hasgrad(xs[i]) && ∇add!(l.as[i], xs[i].gy, l.gy)
  end
end

function add!{T}(a::Number, x::Array{T}, y::Array{T})
  n = length(x)
  for k = 1:n:length(y)
    BLAS.axpy!(n, T(a), pointer(x), stride(x,1), pointer(y,k), stride(y,1))
  end
end

add!{T}(a::Number, x::Number, y::Array{T}) = broadcast!(+, y, y, x)

function ∇add!{T}(a::Number, gx::Array{T}, gy::Array{T})
  n = length(gx)
  for k = 1:n:length(gy)
    BLAS.axpy!(n, T(a), pointer(gy,k), stride(gy,1), pointer(gx), stride(gx,1))
  end
end
