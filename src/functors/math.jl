import Base: +, .+, -, .-, *, .*

type Plus <: Functor; end
type ElemPlus <: Functor; end
type Minus <: Functor; end
type ElemMinus <: Functor; end
type Times <: Functor; end
type ElemTimes <: Functor; end

for op in [:+, :-, :*]
  @eval begin
    $op(x1::Number, x2::Var) = $op(Var(x1), x2)
    $op(x1::Var, x2::Number) = $op(x1, Var(x2))
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
