import Base: +, -, .*, *

function +(x1::Var, x2::Var)
  y = x1.value .+ x2.value
  Var(y, +, [x1,x2], ∇add!)
end

function ∇add!(y::Var)
  x1, x2 = y[1], y[2]
  hasgrad(x1) && ∇add!(1.0, x1.grad, y.grad)
  hasgrad(x2) && ∇add!(1.0, x2.grad, y.grad)
end

function ∇add!{T}(a::Float64, gx::Array{T}, gy::Array{T})
  for offset = 1:length(gx):length(gy)
    BLAS.axpy!(length(gx), T(a), pointer(gy,offset), stride(gy,1), pointer(gx), stride(gx,1))
  end
end

function -(x1::Var, x2::Var)
  y = x1.value .- x2.value
  Var(y, -, [x1,x2], ∇subtract!)
end

function ∇subtract!(y::Var)
  x1, x2 = y[1], y[2]
  hasgrad(x1) && ∇add!(1.0, x1.grad, y.grad)
  hasgrad(x2) && ∇add!(-1.0, x2.grad, y.grad)
end

function *(x1::Var, x2::Var)
  y = x1.value * x2.value
  Var(y, *, [x1,x2], ∇multiply!)
end

function ∇multiply!(y::Var)
  T = eltype(y.value)
  x1, x2 = y[1], y[2]
  hasgrad(x1) && BLAS.gemm!('N', 'T', T(1), y.grad, x2.value, T(1), x1.grad)
  hasgrad(x2) && BLAS.gemm!('T', 'N', T(1), x1.value, y.grad, T(1), x2.grad)
end

function .*(x1::Var, x2::Var)
  y = x1.value .* x2.value
  Var(y, .*, [x1,x2], ∇elemmultiply!)
end

function ∇elemmultiply!(y::Var)
  x1, x2 = y[1], y[2]
  hasgrad(x1) && ∇elemmultiply!(x2.value, x1.grad, y.grad)
  hasgrad(x2) && ∇elemmultiply!(x1.value, x2.grad, y.grad)
end

function ∇elemmultiply!{T,N}(x2::Array{T,N}, gx1::Array{T,N}, gy::Array{T,N})
  @inbounds @simd for i = 1:length(gx1)
    gx1[i] += gy[i] * x2[i]
  end
end
