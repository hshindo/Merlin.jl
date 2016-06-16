import Base: +, .+, -, .-, *, .*

type Plus
  a::Vector
end

type Times; end
type ElemTimes; end

+(x1::Var, x2::Var) = forward(Plus([1,1]), [x1,x2])
+(a::Number, x::Var) = Var([a]) + x
+(x::Var, a::Number) = x + Var([a])

-(x1::Var, x2::Var) = forward(Plus([1,-1]), [x1,x2])
-(x::Var) = forward(Plus([-1]), [x])

*(a::Number, x::Var) = forward(Plus([a]), x)
*(x::Var, a::Number) = a * x
*(x1::Var, x2::Var) = forward(Times(), x1, x2)

.*(x1::Var, x2::Var) = forward(ElemTimes(), [x1,x2])

@compat function (f::Plus){T,N}(xs::Vector{Array{T,N}})
  length(xs) == 1 && return f, (f.a[1] .* xs[1])
  maxi, maxlen = 1, length(xs[1])
  for i = 2:length(xs)
    length(xs[i]) <= maxlen && continue
    maxi = i
    maxlen = length(xs[i])
  end

  y = zeros(xs[maxi])
  for i = 1:length(xs)
    x = xs[i]
    n = length(x)
    for k = 1:n:length(y)
      BLAS.axpy!(n, T(f.a[i]), pointer(x,k), stride(x,1), pointer(y), stride(y,1))
    end
  end
  f, y
end

function backward!{T,N}(f::Plus, xs, gxs::Vector{Array{T,N}}, y, gy::Array{T,N})
  for i = 1:length(gxs)
    gx = gxs[i]
    n = length(gx)
    for k = 1:n:length(gy)
      BLAS.axpy!(n, T(f.a[i]), pointer(gy,k), stride(gy,1), pointer(gx), stride(gx,1))
    end
  end
end

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
  @inbounds @simd for i = 1:length(gx1)
    gx1[i] += gy[i] * x2[i]
  end
end
=#
