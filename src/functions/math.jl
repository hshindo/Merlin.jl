import Base: +, -, .*, *

type Plus
  as::Vector{Number}
end

type Times; end

+(x1::Var, x2::Var) = Plus([1,1])([x1,x2])
+(a::Number, x::Var) = Var(a) + x
+(x::Var, a::Number) = x + Var(a)

-(x1::Var, x2::Var) = Plus([1,-1])([x1,x2])
-(a::Number, x::Var) = Var(a) - x
-(x::Var, a::Number) = x - Var(a)
-(x::Var) = Plus([-1])([x])

*(a::Number, x::Var) = Plus([a])([x])
*(x::Var, a::Number) = a * x

@compat function (f::Plus)(xs::Vector{Var})
  @checkargs f xs

  maxi, maxlen = 1, length(xs[1].value)
  for i = 2:length(xs)
    n = length(xs[i].value)
    n <= maxlen && continue
    maxi = i
    maxlen = n
  end
  y = zeros(xs[maxi].value)
  for i = 1:length(xs)
    add!(f.as[i], xs[i].value, y)
  end

  df(gy) = begin
    for i = 1:length(xs)
      hasgrad(xs[i]) || continue
      ∇add!(f.as[i], xs[i].grad, gy)
    end
  end
  Var(y, df, xs)
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

#=
function plus{T,N}(as::Vector{Float64}, xs::Vector{Array{T,N}})
  y = zeros(xs[maxi])
  for i = 1:length(xs)
    a, x = as[i], xs[i]
    n = length(x)
    for k = 1:n:length(y)
      BLAS.axpy!(n, T(a), pointer(x), stride(x,1), pointer(y,k), stride(y,1))
    end
  end
  y
end
=#

function .*(x1::Var, x2::Var)
  @checkargs .* (x1,x2)
  y = x1.value .* x2.value
  df(gy) = begin
    hasgrad(x1) && ∇elemtimes!(x2.value, x1.grad, gy)
    hasgrad(x2) && ∇elemtimes!(x1.value, x2.grad, gy)
  end
  Var(y, df, [x1,x2])
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

function *(x1::Var, x2::Var)
  @checkargs * (x1,x2)
  y = x1.value * x2.value
  df(gy) = begin
    T = eltype(gy)
    hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), gy, x2.value, T(1), x1.grad)
    hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.value, gy, T(1), x2.grad)
  end
  Var(y, df, [x1,x2])
end

#=
@compat function (f::Times){T}(x1::Matrix{T}, x2::Matrix{T})
  f, BLAS.gemm('N', 'N', T(1), x1, x2)
end

@compat function (f::Times)(x::Array)
  f, f.a * x
end

@compat function (f::ElemTimes){T,N}(xs::Vector{Array{T,N}})
  y = copy(xs[1])
  for i = 2:length(xs)
    broadcast!(.*, y, y, xs[i])
  end
  f, y
end
=#

function optimize(v::Var)

end

#=
for (op,f) in [(:+,:Plus), (:-,:Minus), (:*,:Times)]
  @eval begin
    $op(x1::Number, x2::Var) = forward($f(x1), x2)
    $op(x1::Var, x2::Number) = $op(x2, x1)
    $op(x1::Var, x2::Var) = forward($f(0.0), [x1,x2])
  end
end

@compat function (f::Plus){T<:Array}(xs::Vector{T})
  y = copy(xs[1])
  f.a == nothing || broadcast!(+, y, y, f.a)
  for i = 2:length(xs)
    broadcast!(+, y, y, xs[i])
  end
  f, y
end

function backward!(f::Plus, xs, gxs, y, gy)
  for gx in gxs

  end
end

for (f,op) in [(:Plus,:+), (:ElemPlus,:.+), (:Minus,:-), (:ElemMinus,:.-), (:Times,:*), (:ElemTimes,:.*)]
  @eval begin
    $op(x1::Var, x2::Var) = forward($f(), [x1,x2])
  end
end
-(x::Var) = 0 - x

for (f,op) in [(:Plus,:+), (:ElemPlus,:.+), (:Minus,:-), (:ElemMinus,:.-), (:Times,:*), (:ElemTimes,:.*)]
  @eval begin
    forward{T<:Array}(f::$f, xs::Vector{T}) = f, $op(xs[1], xs[2])
  end
end

for (f,op) in [(:Plus,:+), (:ElemPlus,:.+)]
  @eval begin
    function backward!(f::$f, xs, gxs, y, gy::Array)
      isempty(gxs[1]) || ∇plus!(1.0, gxs[1], gy)
      isempty(gxs[2]) || ∇plus!(1.0, gxs[2], gy)
    end
  end
end

function ∇plus!{T}(a::Float64, gx::Array{T}, gy::Array{T})
  n = min(length(gx), length(gy))
  for offset = 1:n:max(length(gx),length(gy))
    BLAS.axpy!(n, T(a), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end

for (f,op) in [(:Minus,:-), (:ElemMinus,:.-)]
  @eval begin
    function backward!(f::$f, xs, gxs, y, gy::Array)
      isempty(gxs[1]) || ∇plus!(1.0, gxs[1], gy)
      isempty(gxs[2]) || ∇plus!(-1.0, gxs[2], gy)
    end
  end
end

function backward!{T}(f::Times, xs, gxs, y, gy::Array{T})
  x1, x2 = xs[1], xs[2]
  gx1, gx2 = gxs[1], gxs[2]
  isempty(gx1) || BLAS.gemm!('N', 'T', T(1), gy, x2, T(1), gx1)
  isempty(gx2) || BLAS.gemm!('T', 'N', T(1), x1, gy, T(1), gx2)
end

function ∇times!{T<:Matrix}(x1::T, x2::T, gx1::T, gx2::T, gy::T)
  F = eltype(x1)
  isempty(gx1) || BLAS.gemm!('N', 'T', F(1), gy, x2, F(1), gx1)
  isempty(gx2) || BLAS.gemm!('T', 'N', F(1), x1, gy, F(1), gx2)
end

function backward!(f::ElemTimes, xs, gxs, y, gy::Array)
  x1, x2 = xs[1], xs[2]
  gx1, gx2 = gxs[1], gxs[2]
  isempty(gx1) || ∇elemtimes!(x2, gx1, gy)
  isempty(gx2) || ∇elemtimes!(x1, gx2, gy)
end

function ∇elemtimes!{T,N}(x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  one_dims = Int[]
  for (i, n) in enumerate(size(gx1))
    if n == 1
      push!(one_dims, i)
    end
  end

  # TODO: is it possible to avoid memory allocations?
  g = x2 .* gy  # TODO: this line is slow
  s = sum(g, one_dims)
  BLAS.axpy!(T(1), s, gx1)
end
=#
