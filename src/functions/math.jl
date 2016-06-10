import Base: +, .+, -, .-, *, .*

type Plus; end
type ElemPlus; end
type Minus; end
type ElemMinus; end
type Times; end
type ElemTimes; end

for (f,op) in [(:Plus,:+), (:Minus,:-), (:Times,:*)]
  @eval begin
    $op(x1::Number, x2::Var) = $f()([Var(x1),x2])
    $op(x1::Var, x2::Number) = $f()([x1,Var(x2)])
  end
end

for (f,op) in [(:Plus,:+), (:ElemPlus,:.+), (:Minus,:-), (:ElemMinus,:.-), (:Times,:*), (:ElemTimes,:.*)]
  @eval begin
    $op(x1::Var, x2::Var) = $f()([x1,x2])
  end
end
-(x::Var) = 0 - x

for (f,op) in [(:Plus,:+), (:ElemPlus,:.+)]
  @eval begin
    @compat function (f::$f)(args::Vector{Var})
      @checkargs f args
      x1, x2 = args[1], args[2]
      y = $op(x1.value, x2.value)
      function df(gy)
        hasgrad(x1) && ∇plus!(1.0, x1.grad, gy)
        hasgrad(x2) && ∇plus!(1.0, x2.grad, gy)
      end
      Var(y, df, [x1,x2])
    end
  end
end

function ∇plus!{T}(a::Float64, gx::Array{T}, gy::Array{T})
  for offset = 1:length(gx):length(gy)
    BLAS.axpy!(length(gx), T(a), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end

for (f,op) in [(:Minus,:-), (:ElemMinus,:.-)]
  @eval begin
    @compat function (f::$f)(args::Vector{Var})
      @checkargs f args
      x1, x2 = args[1], args[2]
      y = $op(x1.value, x2.value)
      function df(gy)
        hasgrad(x1) && ∇plus!(1.0, x1.grad, gy)
        hasgrad(x2) && ∇plus!(-1.0, x2.grad, gy)
      end
      Var(y, df, [x1,x2])
    end
  end
end

@compat function (f::Times)(args::Vector{Var})
  @checkargs f args
  x1, x2 = args[1], args[2]
  y = x1.value * x2.value
  function df(gy)
    T = eltype(gy)
    hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), gy, x2.value, T(1), x1.grad)
    hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.value, gy, T(1), x2.grad)
  end
  Var(y, df, [x1,x2])
end

@compat function (f::ElemTimes)(args::Vector{Var})
  @checkargs f args
  x1, x2 = args[1], args[2]
  y = x1.value .* x2.value
  function df(gy)
    hasgrad(x1) && ∇elemtimes!(x2.value, x1.grad, gy)
    hasgrad(x2) && ∇elemtimes!(x1.value, x2.grad, gy)
  end
  Var(y, df, [x1,x2])
end

function ∇elemtimes!{T,N}(x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx1)
    gx1[i] += gy[i] * x2[i]
  end
end
