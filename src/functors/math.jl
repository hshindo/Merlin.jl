type Plus; end
type Minus; end
type Times; end
type ElemTimes; end

import Base: +, -, *, .*
+(x1::Var, x2::Var) = forward(Plus(), [x1,x2])
-(x1::Var, x2::Var) = forward(Minus(), [x1,x2])
*(x1::Var, x2::Var) = forward(Times(), [x1,x2])
.*(x1::Var, x2::Var) = forward(ElemTimes(), [x1,x2])

for f in (:+, :-, :.*)
  @eval begin
    $f(x1::Number, x2::Var) = $f(Var([x1]), x2)
    $f(x1::Var, x2::Number) = $f(x1, Var([x2]))
  end
end

forward!(f::Plus, y::Var) = y.value = y[1].value .+ y[2].value
forward!(f::Minus, y::Var) = y.value = y[1].value .- y[2].value
forward!(f::Times, y::Var) = y.value = y[1].value * y[2].value
forward!(f::ElemTimes, y::Var) = y.value = y[1].value .* y[2].value

function backward!(f::Plus, y::Var)
  hasgrad(y[1]) && ∇plus!(1.0, y[1].grad, y.grad)
  hasgrad(y[2]) && ∇plus!(1.0, y[2].grad, y.grad)
end

function backward!(f::Minus, y::Var)
  hasgrad(y[1]) && ∇plus!(1.0, y[1].grad, y.grad)
  hasgrad(y[2]) && ∇plus!(-1.0, y[2].grad, y.grad)
end

function backward!(f::Times, y::Var)
  T = eltype(y.value)
  x1, x2 = y[1], y[2]
  hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), y.grad, x2.value, T(1), x1.grad)
  hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.value, y.grad, T(1), x2.grad)
end

function backward!(f::ElemTimes, y::Var)
  x1, x2 = y[1], y[2]
  hasgrad(x1) && ∇elemtimes!(x2.value, x1.grad, y.grad)
  hasgrad(x2) && ∇elemtimes!(x1.value, x2.grad, y.grad)
end

function ∇plus!{T}(a::Float64, gx::Array{T}, gy::Array{T})
  for offset = 1:length(gx):length(gy)
    BLAS.axpy!(length(gx), T(a), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end

function ∇elemtimes!{T,N}(x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx1)
    gx1[i] += gy[i] * x2[i]
  end
end
