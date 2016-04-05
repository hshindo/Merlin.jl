type Multiply <: Functor
end

function Base.call(f::Multiply, arg1::Variable, arg2::Variable)
  v.value = arg1.value * arg2.value
  v.backward! = () -> begin
    T = eltype(v.value)
    v[1].grad == nothing && (v[1].grad = zeros(v[1].value))
    v[2].grad == nothing && (v[2].grad = zeros(v[2].value))
    gemm!('N', 'T', T(1), v.grad, v[2].value, T(1), v[1].grad)
    gemm!('T', 'N', T(1), v[1].value, v.grad, T(1), v[2].grad)
  end
end

function multiply!(f::Multiply, x1::Array, x2::Array, y::Array)
  gemm!('N', 'N', T(1), x1, x2, T(-1), y)
end

import Base.*
*(v1::Variable, v2::Variable) = Multiply()(v1, v2)

"""
## GEMM
BLAS gemm function.
This is in-place operation.
"""
type GEMM <: Functor
end

function call(f::GEMM, y::Variable, x1::Variable, x2::Variable)
  T = eltype(y)
  gemm!('N', 'N', T(1), x1.value, x2.value, T(1), y.value)
  backward! = gy -> begin
    ()
  end
  Variable(f, [y,x1,x2], y.value, backward!)
end
