import Base: +, .+, -, .-, *, .*

type Plus; end
type ElemPlus; end
type Minus; end
type ElemMinus; end
type Times; end
type ElemTimes; end

for op in [:+, :-, :*]
  @eval begin
    $op(x1::Number, x2::Var) = $op(Var(x1), x2)
    $op(x1::Var, x2::Number) = $op(x1, Var(x2))
  end
end

for (op,t) in [(:+,:Plus), (:.+,:ElemPlus)]
  @eval begin
    $op(x1::Var, x2::Var) = forward($t(), [x1,x2])

    @compat function (f::$t)(args::Vector{Var})
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

for (op,t) in [(:-,:Minus), (:.-,:ElemMinus)]
  @eval begin
    $op(x1::Var, x2::Var) = forward($t(), [x1,x2])
    @compat function (f::$t)(args::Vector{Var})
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
  x1, x2 = args[1], args[2]
  y = x1.value * x2.value
  function df(gy)
    T = eltype(gy)
    hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), gy, x2.value, T(1), x1.grad)
    hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.value, gy, T(1), x2.grad)
  end
  Var(y, df, [x1,x2])
end
*(x1::Var, x2::Var) = forward(Times(), [x1,x2])

@compat function (f::ElemTimes)(args::Vector{Var})
  x1, x2 = args[1], args[2]
  y = x1.value .* x2.value
  function df(gy)
    hasgrad(x1) && ∇elemtimes!(x2.value, x1.grad, gy)
    hasgrad(x2) && ∇elemtimes!(x1.value, x2.grad, gy)
  end
  Var(y, df, [x1,x2])
end
.*(x1::Var, x2::Var) = forward(ElemTimes(), [x1,x2])

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
