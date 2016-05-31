type Add <: Functor
end
type Subtract <: Functor
end
type Multiply <: Functor
end
type ElemMultiply <: Functor
end

import Base: +, -, *, .*
for (f,op) in [(Add,:+), (Subtract,:-), (Multiply,:*), (ElemMultiply,:.*)]
  @eval begin
    $op(x1::Var, x2::Var) = forward0($f(), [x1,x2])
    $op(x1::Number, x2::Var) = forward0($f(), [Var([x1]),x2])
    $op(x1::Var, x2::Number) = forward0($f(), [x1,Var([x2])])
  end
end

function forward(f::Add, args::Vector{Var})
  x1, x2 = args[1], args[2]
  y = x1.val + x2.val
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x1) && BLAS.axpy!(T(1), gy, x1.grad)
    hasgrad(x2) && BLAS.axpy!(T(1), gy, x2.grad)
  end
  Var(y, nothing, f, args, backward!)
end

#=
function backward!(f::Add, x1::Var, x2::Var, y, gy)
  hasgrad(x1) && BLAS.axpy!(T(1), gy, x1.grad)
  gx, gy = x.grad, y.grad
  T = eltype(x.val)
  for offset = 1:length(gx):length(gy)
    BLAS.axpy!(length(gx), T(a), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end
=#

function forward(f::Multiply, args::Vector{Var})
  x1, x2 = args[1], args[2]
  y = x1.val * x2.val
  backward! = gy -> begin
    T = eltype(gy)
    hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), gy, x2.val, T(1), x1.grad)
    hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.val, gy, T(1), x2.grad)
  end
  Var(y, nothing, f, args, backward!)
end

function forward(f::ElemMultiply, args::Vector{Var})
  x1, x2 = args[1], args[2]
  y = x1.val .* x2.val
  backward! = gy -> begin
    hasgrad(x1) && ∇elemmult!(x2.val, x1.grad, gy)
    hasgrad(x2) && ∇elemmult!(x1.val, x2.grad, gy)
  end
  Var(y, nothing, f, args, backward!)
end

function ∇elemmult!{T,N}(x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx1)
    gx1[i] += gy[i] * x2[i]
  end
end
